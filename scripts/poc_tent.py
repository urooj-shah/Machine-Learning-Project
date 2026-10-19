"""PoC: TENT-style entropy minimization TTA on decoder BN layers.

Unlike AdaBN (recompute running stats), TENT backprops entropy loss
through BN affine parameters (gamma/beta) to encourage confident predictions.

Test on all PolypGen sites, compare with source-only and AdaBN baselines.
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
from eval import GRID, N_PATCHES, dice_score, load_site_features, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks=None):
        self.tokens = patch_tokens
        self.masks = masks

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x))
        if self.masks is not None:
            return x, torch.from_numpy(self.masks[idx]).unsqueeze(0)
        return x


def load_masks_128(mask_paths):
    masks = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
        masks.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(masks)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder(train_tokens, train_masks, val_tokens, val_masks,
                  n_epochs=80, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatureDataset(train_tokens, train_masks)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
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
        vds = FeatureDataset(val_tokens, val_masks)
        vdl = DataLoader(vds, batch_size=32, shuffle=False, num_workers=0)
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
def eval_dice_128(model, tokens, gt_128):
    model.eval()
    dices = []
    for i in range(0, len(tokens), 64):
        batch = tokens[i:i + 64]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        logits = model(x).cpu().numpy()[:, 0]
        for j in range(len(batch)):
            pred = (sigmoid(logits[j]) >= 0.5).astype(np.uint8)
            gt = (gt_128[i + j] >= 0.5).astype(np.uint8)
            dices.append(dice_score(pred, gt))
    return np.array(dices)


def entropy_loss(logits):
    """Per-pixel entropy of sigmoid predictions, averaged."""
    probs = torch.sigmoid(logits)
    ent = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    return ent.mean()


def tent_adapt(model, target_tokens, n_steps=10, lr=1e-3, batch_size=32):
    """TENT: adapt only BN affine params (weight/bias) via entropy minimization."""
    adapted = copy.deepcopy(model)

    # Collect BN affine parameters only
    bn_params = []
    for m in adapted.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            bn_params.extend([m.weight, m.bias])
            # Reset running stats and use batch stats during adaptation
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None

    # Freeze everything else
    for p in adapted.parameters():
        p.requires_grad_(False)
    for p in bn_params:
        p.requires_grad_(True)

    opt = torch.optim.Adam(bn_params, lr=lr)

    ds = FeatureDataset(target_tokens)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for step in range(n_steps):
        for x in dl:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(DEVICE)
            adapted.train()
            logits = adapted(x)
            loss = entropy_loss(logits)
            opt.zero_grad()
            loss.backward()
            opt.step()

    adapted.eval()
    return adapted


@torch.no_grad()
def adabn_adapt(model, target_tokens):
    """AdaBN baseline: recompute running stats from target data."""
    adapted = copy.deepcopy(model)
    for m in adapted.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None
    adapted.train()
    for i in range(0, len(target_tokens), 64):
        batch = target_tokens[i:i + 64]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        adapted(x)
    adapted.eval()
    return adapted


def main():
    t0 = time.time()

    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])

    print("Training source decoder ...", flush=True)
    model, vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  val_dice={vd:.4f}", flush=True)

    # First: sweep TENT hyperparams on Kvasir val to find best settings
    print("\n=== TENT hyperparameter sweep (Kvasir val) ===", flush=True)
    val_tokens = kvasir["patch_tokens"][val_idx]
    val_gt = kvasir_m128[val_idx]

    baseline_dice = eval_dice_128(model, val_tokens, val_gt).mean()
    print(f"  Source-only: {baseline_dice:.4f}")

    best_cfg = None
    best_val_dice = 0
    for n_steps in [1, 3, 5, 10, 20]:
        for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
            adapted = tent_adapt(model, val_tokens, n_steps=n_steps, lr=lr)
            d = eval_dice_128(adapted, val_tokens, val_gt).mean()
            tag = "*" if d > best_val_dice else " "
            if d > best_val_dice:
                best_val_dice = d
                best_cfg = (n_steps, lr)
            print(f"  {tag} steps={n_steps:>2} lr={lr:.0e}: {d:.4f} (delta={d - baseline_dice:+.4f})")

    print(f"\n  Best: steps={best_cfg[0]}, lr={best_cfg[1]:.0e} -> {best_val_dice:.4f}")

    # Now evaluate on all target sites with best config
    print(f"\n{'='*90}")
    print(f"TABLE: TENT vs AdaBN vs Source-only (steps={best_cfg[0]}, lr={best_cfg[1]:.0e})")
    print(f"{'='*90}")
    print(f"{'site':>15} {'n':>5} {'source':>10} {'adabn':>10} {'tent':>10} "
          f"{'d_adabn':>10} {'d_tent':>10}")
    print("-" * 90)

    sites = ["kvasir_val", "polypgen_C1", "polypgen_C2", "polypgen_C3",
             "polypgen_C4", "polypgen_C5", "polypgen_C6"]

    for site_name in sites:
        if site_name == "kvasir_val":
            tokens = val_tokens
            gt = val_gt
        else:
            sd = load_site_features(FEAT_DIR, site_name)
            tokens = sd["patch_tokens"]
            gt = load_masks_128(sd["mask_paths"])

        d_src = eval_dice_128(model, tokens, gt)
        pos = np.array([(g >= 0.5).sum() > 0 for g in gt])

        # AdaBN
        m_adabn = adabn_adapt(model, tokens)
        d_adabn = eval_dice_128(m_adabn, tokens, gt)

        # TENT
        m_tent = tent_adapt(model, tokens, n_steps=best_cfg[0], lr=best_cfg[1])
        d_tent = eval_dice_128(m_tent, tokens, gt)

        src_pos = d_src[pos].mean() if pos.sum() > 0 else float("nan")
        abn_pos = d_adabn[pos].mean() if pos.sum() > 0 else float("nan")
        tnt_pos = d_tent[pos].mean() if pos.sum() > 0 else float("nan")

        print(f"{site_name:>15} {pos.sum():>5} {src_pos:>10.4f} {abn_pos:>10.4f} "
              f"{tnt_pos:>10.4f} {abn_pos - src_pos:>+10.4f} {tnt_pos - src_pos:>+10.4f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
