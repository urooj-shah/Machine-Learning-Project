"""PoC: Multi-resolution consistency self-training.

Novel idea: train decoders at 224 and 448, use agreement between them
as a confidence signal for pseudo-labels, fine-tune 448 decoder on
agreement-filtered pseudo-labels.

Two independent resolutions agreeing is a stronger signal than
a single model's softmax confidence.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import dice_score, load_site_features, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR_224 = ROOT / "features" / "dinov2_vits14_224"
FEAT_DIR_448 = ROOT / "features" / "dinov2_vits14_448"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

GRID_224 = 16
GRID_448 = 32
MASK_224 = 128
MASK_448 = 256


def load_masks(mask_paths, size):
    masks = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((size, size), Image.NEAREST)
        masks.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(masks)


def load_448(name):
    d = np.load(FEAT_DIR_448 / f"{name}.npz")
    return {
        "patch_tokens": d["patch_tokens"].astype(np.float32),
        "filenames": d["filenames"],
        "mask_paths": d["mask_paths"],
    }


class FeatDS(Dataset):
    def __init__(self, tokens, masks, grid):
        self.tokens = tokens
        self.masks = masks
        self.grid = grid

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder(tokens, masks, val_tokens, val_masks, grid, mask_size,
                  n_epochs=80, lr=1e-3, batch_size=32, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatDS(tokens, masks, grid)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = PatchDecoder(in_dim=384).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss()
    best_dice, best_state = 0.0, None
    for _ in range(n_epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = bce(model(x), y) + dice_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        vds = FeatDS(val_tokens, val_masks, grid)
        vdl = DataLoader(vds, batch_size=batch_size, shuffle=False, num_workers=0)
        td = 0
        with torch.no_grad():
            for x, y in vdl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                inter = (p * y).sum(dim=(1, 2, 3))
                union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                td += ((2 * inter + 1) / (union + 1)).sum().item()
        vd = td / len(vds)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best_dice


@torch.no_grad()
def predict_probs(model, tokens, grid, mask_size):
    """Return probability maps at mask_size resolution."""
    model.eval()
    all_probs = []
    bs = 32 if grid == 16 else 16
    for i in range(0, len(tokens), bs):
        batch = tokens[i:i + bs]
        x = np.stack([t.reshape(grid, grid, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        logits = model(x).cpu().numpy()[:, 0]
        all_probs.append(sigmoid(logits))
    return np.concatenate(all_probs)


def eval_dice(probs, gt, threshold=0.5):
    dices = []
    pos_mask = []
    for i in range(len(probs)):
        pred = (probs[i] >= threshold).astype(np.uint8)
        g = (gt[i] >= 0.5).astype(np.uint8)
        is_pos = g.sum() > 0
        pos_mask.append(is_pos)
        dices.append(dice_score(pred, g))
    return np.array(dices), np.array(pos_mask)


def finetune_on_pseudolabels(model, tokens, pseudo_masks, grid, mask_size,
                              n_steps=50, lr=1e-3, batch_size=16, seed=42):
    """Fine-tune decoder on pseudo-label masks."""
    torch.manual_seed(seed)
    adapted = copy.deepcopy(model)
    ds = FeatDS(tokens, pseudo_masks, grid)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    adapted.train()
    step = 0
    while step < n_steps:
        for x, y in dl:
            if step >= n_steps:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = bce(adapted(x), y) + dice_loss(adapted(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    adapted.eval()
    return adapted


def main():
    t0 = time.time()

    print("Loading data ...", flush=True)
    kvasir_224 = load_site_features(FEAT_DIR_224, "kvasir")
    kvasir_448 = load_448("kvasir")

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns_224 = list(kvasir_224["filenames"])
    fns_448 = list(kvasir_448["filenames"])
    train_idx_224 = np.array([fns_224.index(fn) for fn in split["train"]])
    val_idx_224 = np.array([fns_224.index(fn) for fn in split["val"]])
    train_idx_448 = np.array([fns_448.index(fn) for fn in split["train"]])
    val_idx_448 = np.array([fns_448.index(fn) for fn in split["val"]])

    masks_224_train = load_masks(kvasir_224["mask_paths"][train_idx_224], MASK_224)
    masks_224_val = load_masks(kvasir_224["mask_paths"][val_idx_224], MASK_224)
    masks_448_train = load_masks(kvasir_448["mask_paths"][train_idx_448], MASK_448)
    masks_448_val = load_masks(kvasir_448["mask_paths"][val_idx_448], MASK_448)

    # Train source decoders at both resolutions
    print("Training 224 decoder ...", flush=True)
    model_224, vd_224 = train_decoder(
        kvasir_224["patch_tokens"][train_idx_224], masks_224_train,
        kvasir_224["patch_tokens"][val_idx_224], masks_224_val,
        GRID_224, MASK_224, batch_size=32, seed=SEED)
    print(f"  val_dice={vd_224:.4f}", flush=True)

    print("Training 448 decoder ...", flush=True)
    model_448, vd_448 = train_decoder(
        kvasir_448["patch_tokens"][train_idx_448], masks_448_train,
        kvasir_448["patch_tokens"][val_idx_448], masks_448_val,
        GRID_448, MASK_448, batch_size=16, seed=SEED)
    print(f"  val_dice={vd_448:.4f}", flush=True)

    # Evaluate on all sites
    sites = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
             "polypgen_C4", "polypgen_C5", "polypgen_C6"]

    print(f"\n{'='*110}")
    print("TABLE: MULTI-RESOLUTION CONSISTENCY SELF-TRAINING")
    print(f"{'='*110}")
    print(f"{'site':>15} {'n_pos':>6} {'src_224':>8} {'src_448':>8} "
          f"{'agree%':>7} {'pl_448':>7} {'consist':>8} {'d_pl':>8} {'d_con':>8}")
    print("-" * 110)

    for site_name in sites:
        sd_224 = load_site_features(FEAT_DIR_224, site_name)
        sd_448 = load_448(site_name)

        tokens_224 = sd_224["patch_tokens"]
        tokens_448 = sd_448["patch_tokens"]
        gt_224 = load_masks(sd_224["mask_paths"], MASK_224)
        gt_448 = load_masks(sd_448["mask_paths"], MASK_448)

        # Source-only predictions
        probs_224 = predict_probs(model_224, tokens_224, GRID_224, MASK_224)
        probs_448 = predict_probs(model_448, tokens_448, GRID_448, MASK_448)

        d_224, pos_224 = eval_dice(probs_224, gt_224)
        d_448, pos_448 = eval_dice(probs_448, gt_448)

        # Build agreement mask at 448 resolution
        # Upsample 224 predictions to 448 size for comparison
        preds_224_bin = (probs_224 >= 0.5).astype(np.float32)
        preds_448_bin = (probs_448 >= 0.5).astype(np.float32)

        # Upsample 224 preds to 256x256 (448's mask size)
        preds_224_up = np.stack([
            np.array(Image.fromarray(p, mode="F").resize((MASK_448, MASK_448), Image.NEAREST))
            for p in preds_224_bin
        ])

        # Agreement: both predict same class
        agreement = (preds_224_up == preds_448_bin).astype(np.float32)
        agree_rate = agreement.mean()

        # Method A: plain pseudo-label self-training (448 predictions as labels)
        pseudo_masks_plain = preds_448_bin
        model_pl = finetune_on_pseudolabels(
            model_448, tokens_448, pseudo_masks_plain,
            GRID_448, MASK_448, n_steps=50, seed=SEED)
        probs_pl = predict_probs(model_pl, tokens_448, GRID_448, MASK_448)
        d_pl, _ = eval_dice(probs_pl, gt_448)

        # Method B: consistency-filtered pseudo-labels
        # Only use pixels where both resolutions agree
        # For disagreeing pixels, use soft label 0.5 (ignore via weighting)
        # Simpler: mask out disagreeing pixels in the loss
        # Implementation: set pseudo-label = 448 prediction where agreement,
        #                 = -1 (ignore) where disagreement
        # For fine-tuning, we'll use a weighted approach:
        # high weight for agreement pixels, low for disagreement
        consist_masks = np.where(agreement > 0.5, preds_448_bin, 0.5 * np.ones_like(preds_448_bin))
        model_con = finetune_on_consistency(
            model_448, tokens_448, preds_448_bin, agreement,
            GRID_448, MASK_448, n_steps=50, seed=SEED)
        probs_con = predict_probs(model_con, tokens_448, GRID_448, MASK_448)
        d_con, _ = eval_dice(probs_con, gt_448)

        src_224 = d_224[pos_224].mean() if pos_224.sum() > 0 else float("nan")
        src_448 = d_448[pos_448].mean() if pos_448.sum() > 0 else float("nan")
        pl_dice = d_pl[pos_448].mean() if pos_448.sum() > 0 else float("nan")
        con_dice = d_con[pos_448].mean() if pos_448.sum() > 0 else float("nan")

        print(f"{site_name:>15} {pos_448.sum():>6} {src_224:>8.4f} {src_448:>8.4f} "
              f"{agree_rate:>7.1%} {pl_dice:>7.4f} {con_dice:>8.4f} "
              f"{pl_dice - src_448:>+8.4f} {con_dice - src_448:>+8.4f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


class WeightedFeatDS(Dataset):
    def __init__(self, tokens, pseudo_masks, weights, grid):
        self.tokens = tokens
        self.masks = pseudo_masks
        self.weights = weights
        self.grid = grid

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.masks[idx]).unsqueeze(0),
                torch.from_numpy(self.weights[idx]).unsqueeze(0))


def finetune_on_consistency(model, tokens, pseudo_masks, agreement_masks,
                             grid, mask_size, n_steps=50, lr=1e-3,
                             batch_size=16, seed=42):
    """Fine-tune using consistency-weighted loss.

    High weight (1.0) for pixels where 224 and 448 agree.
    Low weight (0.1) for pixels where they disagree.
    """
    torch.manual_seed(seed)
    adapted = copy.deepcopy(model)
    weights = np.where(agreement_masks > 0.5, 1.0, 0.1).astype(np.float32)
    ds = WeightedFeatDS(tokens, pseudo_masks, weights, grid)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    adapted.train()
    step = 0
    while step < n_steps:
        for x, y, w in dl:
            if step >= n_steps:
                break
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            logits = adapted(x)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y, weight=w)
            probs = torch.sigmoid(logits)
            smooth = 1.0
            inter = (probs * y * w).sum(dim=(1, 2, 3))
            union = (probs * w).sum(dim=(1, 2, 3)) + (y * w).sum(dim=(1, 2, 3))
            dl_loss = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
            loss = bce + dl_loss
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    adapted.eval()
    return adapted


if __name__ == "__main__":
    main()
