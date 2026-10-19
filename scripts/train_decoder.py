"""Train lightweight decoder on Kvasir frozen DINOv2 patch features.

Loads pre-extracted patch tokens, trains a small conv decoder for
polyp segmentation, evaluates on val split. Saves checkpoint.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from patch_decoder import PatchDecoder

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CKPT_DIR = ROOT / "checkpoints"

GRID = 16
MASK_SIZE = 128
SEED = 42


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(384, GRID, GRID)
        y = self.masks[idx]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


def load_masks_128(mask_paths):
    """Load and resize masks to 128x128, binarize."""
    masks = []
    for p in tqdm(mask_paths, desc="Loading masks", leave=False):
        m = Image.open(str(p)).convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
        m = (np.array(m, dtype=np.float32) >= 128).astype(np.float32)
        masks.append(m)
    return np.stack(masks)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def compute_dice(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    smooth = 1.0
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + smooth) / (union + smooth)).mean().item()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load features and split
    print("Loading Kvasir features ...")
    data = np.load(FEAT_DIR / "kvasir.npz")
    patch_tokens = data["patch_tokens"].astype(np.float32)
    filenames = list(data["filenames"])
    mask_paths = data["mask_paths"]

    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_idx = [filenames.index(fn) for fn in split["train"]]
    val_idx = [filenames.index(fn) for fn in split["val"]]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Load masks at 128x128
    print("Loading masks ...")
    masks_128 = load_masks_128(mask_paths)

    # Source prior (natural pixel-level fg rate in training data)
    pi_source = masks_128[train_idx].mean()
    print(f"Source pixel-level fg rate: {pi_source:.4f}")

    # Datasets
    train_ds = FeatureDataset(patch_tokens[train_idx], masks_128[train_idx])
    val_ds = FeatureDataset(patch_tokens[val_idx], masks_128[val_idx])
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Model
    model = PatchDecoder(in_dim=384).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Decoder parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    bce = nn.BCEWithLogitsLoss()

    # Training
    best_val_dice = 0
    best_epoch = 0

    print("\nTraining ...")
    t0 = time.time()

    for epoch in range(80):
        model.train()
        train_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)
        scheduler.step()

        # Val
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_dice += compute_dice(logits, y) * len(x)
        val_dice /= len(val_ds)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "pi_source": pi_source,
            }, CKPT_DIR / "decoder_best.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={train_loss:.4f}  val_dice={val_dice:.4f}  best={best_val_dice:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s")
    print(f"Best val Dice: {best_val_dice:.4f} at epoch {best_epoch+1}")
    print(f"Checkpoint: {CKPT_DIR / 'decoder_best.pt'}")
    print(f"Source prior (pi_s): {pi_source:.4f}")


if __name__ == "__main__":
    main()
