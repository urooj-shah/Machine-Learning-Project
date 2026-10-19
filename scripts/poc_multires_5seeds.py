"""Multi-resolution consistency self-training: 5-seed validation.

Trains 5 decoder pairs (224 + 448), runs plain pseudo-label and
consistency-filtered self-training on all 6 PolypGen sites.
Reports mean ± std across seeds.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT_DIR = ROOT / "outputs" / "08_controls"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]

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
        self.tokens, self.masks, self.grid = tokens, masks, grid
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


class WeightedFeatDS(Dataset):
    def __init__(self, tokens, masks, weights, grid):
        self.tokens, self.masks, self.weights, self.grid = tokens, masks, weights, grid
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.masks[idx]).unsqueeze(0),
                torch.from_numpy(self.weights[idx]).unsqueeze(0))


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder(tokens, masks, val_tokens, val_masks, grid,
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
def predict_probs(model, tokens, grid):
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


def eval_dice_pos(probs, gt):
    dices = []
    for i in range(len(probs)):
        g = (gt[i] >= 0.5).astype(np.uint8)
        if g.sum() == 0:
            continue
        pred = (probs[i] >= 0.5).astype(np.uint8)
        dices.append(dice_score(pred, g))
    return np.mean(dices) if dices else float("nan")


def finetune_plain(model, tokens, pseudo_masks, grid,
                   n_steps=50, lr=1e-3, batch_size=16, seed=42):
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
            logits = adapted(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    adapted.eval()
    return adapted


def finetune_consistency(model, tokens, pseudo_masks, agreement,
                         grid, n_steps=50, lr=1e-3, batch_size=16, seed=42):
    torch.manual_seed(seed)
    adapted = copy.deepcopy(model)
    weights = np.where(agreement > 0.5, 1.0, 0.1).astype(np.float32)
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


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    sites = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
             "polypgen_C4", "polypgen_C5", "polypgen_C6"]

    # Preload all site data
    site_data = {}
    for site_name in sites:
        sd_224 = load_site_features(FEAT_DIR_224, site_name)
        sd_448 = load_448(site_name)
        gt_224 = load_masks(sd_224["mask_paths"], MASK_224)
        gt_448 = load_masks(sd_448["mask_paths"], MASK_448)
        site_data[site_name] = {
            "tokens_224": sd_224["patch_tokens"],
            "tokens_448": sd_448["patch_tokens"],
            "gt_224": gt_224, "gt_448": gt_448,
        }

    all_rows = []

    for seed in SEEDS:
        ts = time.time()
        print(f"\n=== Seed {seed} ===", flush=True)

        # Train decoder pair
        model_224, vd_224 = train_decoder(
            kvasir_224["patch_tokens"][train_idx_224], masks_224_train,
            kvasir_224["patch_tokens"][val_idx_224], masks_224_val,
            GRID_224, batch_size=32, seed=seed)

        model_448, vd_448 = train_decoder(
            kvasir_448["patch_tokens"][train_idx_448], masks_448_train,
            kvasir_448["patch_tokens"][val_idx_448], masks_448_val,
            GRID_448, batch_size=16, seed=seed)

        print(f"  224 val={vd_224:.4f}  448 val={vd_448:.4f}", flush=True)

        for site_name in sites:
            sd = site_data[site_name]
            tokens_224 = sd["tokens_224"]
            tokens_448 = sd["tokens_448"]
            gt_448 = sd["gt_448"]

            # Source-only predictions
            probs_224 = predict_probs(model_224, tokens_224, GRID_224)
            probs_448 = predict_probs(model_448, tokens_448, GRID_448)

            src_224 = eval_dice_pos(probs_224, sd["gt_224"])
            src_448 = eval_dice_pos(probs_448, gt_448)

            # Agreement mask
            preds_224_bin = (probs_224 >= 0.5).astype(np.float32)
            preds_448_bin = (probs_448 >= 0.5).astype(np.float32)
            preds_224_up = np.stack([
                np.array(Image.fromarray(p, mode="F").resize((MASK_448, MASK_448), Image.NEAREST))
                for p in preds_224_bin
            ])
            agreement = (preds_224_up == preds_448_bin).astype(np.float32)

            # Plain pseudo-label
            m_pl = finetune_plain(model_448, tokens_448, preds_448_bin,
                                  GRID_448, n_steps=50, seed=seed)
            pl_dice = eval_dice_pos(predict_probs(m_pl, tokens_448, GRID_448), gt_448)

            # Consistency-filtered
            m_con = finetune_consistency(model_448, tokens_448, preds_448_bin,
                                         agreement, GRID_448, n_steps=50, seed=seed)
            con_dice = eval_dice_pos(predict_probs(m_con, tokens_448, GRID_448), gt_448)

            all_rows.append({
                "seed": seed, "site": site_name,
                "src_224": src_224, "src_448": src_448,
                "plain_pl": pl_dice, "consistency": con_dice,
                "d_pl": pl_dice - src_448, "d_con": con_dice - src_448,
                "agree_pct": agreement.mean(),
            })
            print(f"  {site_name}: 448={src_448:.4f} pl={pl_dice:.4f}({pl_dice-src_448:+.4f}) "
                  f"con={con_dice:.4f}({con_dice-src_448:+.4f})", flush=True)

        print(f"  Seed time: {time.time() - ts:.0f}s", flush=True)

    # Aggregate
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "ctrl_multires_consistency.csv", index=False)

    print(f"\n{'='*120}")
    print("TABLE: MULTI-RESOLUTION SELF-TRAINING (5 seeds, positives-only Dice)")
    print(f"{'='*120}")
    print(f"{'site':>15} {'src_224':>14} {'src_448':>14} {'plain_pl':>14} "
          f"{'consistency':>14} {'d_pl':>10} {'d_con':>10}")
    print("-" * 120)

    for site_name in sites:
        sd = df[df.site == site_name]
        print(f"{site_name:>15} "
              f"{sd.src_224.mean():>6.4f}±{sd.src_224.std():>.4f} "
              f"{sd.src_448.mean():>6.4f}±{sd.src_448.std():>.4f} "
              f"{sd.plain_pl.mean():>6.4f}±{sd.plain_pl.std():>.4f} "
              f"{sd.consistency.mean():>6.4f}±{sd.consistency.std():>.4f} "
              f"{sd.d_pl.mean():>+10.4f} {sd.d_con.mean():>+10.4f}")

    # Count wins
    for method, col in [("plain_pl", "d_pl"), ("consistency", "d_con")]:
        wins = 0
        for site_name in sites:
            sd = df[df.site == site_name]
            if sd[col].mean() > 0:
                wins += 1
        print(f"\n  {method}: positive on {wins}/6 sites (mean delta)")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
