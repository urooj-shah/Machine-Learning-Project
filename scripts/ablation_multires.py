"""Ablation study for multi-resolution consistency self-training.

Variants:
  0  source-only 448
  1  ensemble 224+448 (average probs, no fine-tuning)
  2  self-training on 448 decoder's own pseudo-labels, uniform weight
  3  ours: agreement-weighted pseudo-labels (224 vs 448)
  4  agreement between two 448 decoders (different seeds)

All evaluated at original mask resolution via src/eval.py.
5 seeds, 6 PolypGen centers.
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
from eval import dice_score, load_gt_mask, load_site_features, sigmoid, upsample_probs
from patch_decoder import PatchDecoder

FEAT_DIR_224 = ROOT / "features" / "dinov2_vits14_224"
FEAT_DIR_448 = ROOT / "features" / "dinov2_vits14_448"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "11_ablations"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]

GRID_224, GRID_448 = 16, 32
MASK_224, MASK_448 = 128, 256
FT_STEPS = 50
FT_LR = 1e-3
FT_BS = 16


# ── Data helpers ──────────────────────────────────────────────────

def load_masks(mask_paths, size):
    out = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((size, size), Image.NEAREST)
        out.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(out)


def load_448(name):
    d = np.load(FEAT_DIR_448 / f"{name}.npz")
    return {
        "patch_tokens": d["patch_tokens"].astype(np.float32),
        "filenames": d["filenames"],
        "mask_paths": d["mask_paths"],
        "orig_sizes": d["orig_sizes"],
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


class WeightedDS(Dataset):
    def __init__(self, tokens, masks, weights, grid):
        self.tokens, self.masks, self.weights, self.grid = tokens, masks, weights, grid
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(self.masks[idx]).unsqueeze(0),
                torch.from_numpy(self.weights[idx]).unsqueeze(0))


# ── Training / inference ──────────────────────────────────────────

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    s = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + s) / (union + s)).mean()


def train_decoder(tokens, masks, val_tokens, val_masks, grid,
                  n_epochs=80, lr=1e-3, bs=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = FeatDS(tokens, masks, grid)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
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
        vdl = DataLoader(vds, batch_size=bs, shuffle=False, num_workers=0)
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
def get_logits(model, tokens, grid):
    model.eval()
    out = []
    bs = 32 if grid == 16 else 16
    for i in range(0, len(tokens), bs):
        batch = tokens[i:i + bs]
        x = np.stack([t.reshape(grid, grid, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_at_orig(probs, mask_paths, orig_sizes, out_size):
    """Evaluate at original mask resolution. Returns per-image metrics."""
    results = []
    for i in range(len(probs)):
        ow, oh = int(orig_sizes[i][0]), int(orig_sizes[i][1])
        prob_map = np.asarray(probs[i], dtype=np.float32)
        prob_up = np.array(
            Image.fromarray(prob_map, mode="F").resize((ow, oh), Image.BILINEAR))
        pred = (prob_up >= 0.5).astype(np.uint8)
        gt = load_gt_mask(mask_paths[i])
        gt_b = gt.astype(bool)
        pred_b = pred.astype(bool)
        is_pos = gt_b.sum() > 0
        tp = int((pred_b & gt_b).sum())
        fp = int((pred_b & ~gt_b).sum())
        fn = int((~pred_b & gt_b).sum())
        results.append({
            "dice": dice_score(pred, gt),
            "is_pos": is_pos,
            "fp_on_empty": bool((not is_pos) and pred_b.sum() > 0),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return results


def aggregate(per_image):
    pos = [r for r in per_image if r["is_pos"]]
    neg = [r for r in per_image if not r["is_pos"]]
    all_dices = [r["dice"] for r in per_image]
    pos_dices = [r["dice"] for r in pos]
    tp = sum(r["tp"] for r in per_image)
    fp = sum(r["fp"] for r in per_image)
    fn = sum(r["fn"] for r in per_image)
    pooled = float(2 * tp / max(2 * tp + fp + fn, 1))
    fp_rate = sum(1 for r in neg if r["fp_on_empty"]) / max(len(neg), 1) if neg else 0.0
    return {
        "dice_pos": float(np.mean(pos_dices)) if pos_dices else float("nan"),
        "dice_all": float(np.mean(all_dices)),
        "dice_pooled": pooled,
        "fp_rate": fp_rate,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


# ── Fine-tuning helpers ──────────────────────────────────────────

def finetune_uniform(model, tokens, pseudo_masks, grid, seed=42):
    torch.manual_seed(seed)
    adapted = copy.deepcopy(model)
    ds = FeatDS(tokens, pseudo_masks, grid)
    dl = DataLoader(ds, batch_size=FT_BS, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(adapted.parameters(), lr=FT_LR)
    bce_fn = nn.BCEWithLogitsLoss()
    adapted.train()
    step = 0
    while step < FT_STEPS:
        for x, y in dl:
            if step >= FT_STEPS:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = adapted(x)
            loss = bce_fn(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    adapted.eval()
    return adapted


def finetune_weighted(model, tokens, pseudo_masks, weights, grid, seed=42):
    torch.manual_seed(seed)
    adapted = copy.deepcopy(model)
    ds = WeightedDS(tokens, pseudo_masks, weights, grid)
    dl = DataLoader(ds, batch_size=FT_BS, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(adapted.parameters(), lr=FT_LR)
    adapted.train()
    step = 0
    while step < FT_STEPS:
        for x, y, w in dl:
            if step >= FT_STEPS:
                break
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            logits = adapted(x)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y, weight=w)
            probs = torch.sigmoid(logits)
            s = 1.0
            inter = (probs * y * w).sum(dim=(1, 2, 3))
            union = (probs * w).sum(dim=(1, 2, 3)) + (y * w).sum(dim=(1, 2, 3))
            dl_ = 1 - ((2 * inter + s) / (union + s)).mean()
            loss = bce + dl_
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    adapted.eval()
    return adapted


# ── Main ──────────────────────────────────────────────────────────

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

    masks_224_tr = load_masks(kvasir_224["mask_paths"][train_idx_224], MASK_224)
    masks_224_va = load_masks(kvasir_224["mask_paths"][val_idx_224], MASK_224)
    masks_448_tr = load_masks(kvasir_448["mask_paths"][train_idx_448], MASK_448)
    masks_448_va = load_masks(kvasir_448["mask_paths"][val_idx_448], MASK_448)

    sites = [f"polypgen_C{i}" for i in range(1, 7)]

    # Preload site data
    site_data = {}
    for sn in sites:
        s224 = load_site_features(FEAT_DIR_224, sn)
        s448 = load_448(sn)
        site_data[sn] = {
            "tok224": s224["patch_tokens"], "tok448": s448["patch_tokens"],
            "mp224": s224["mask_paths"], "mp448": s448["mask_paths"],
            "os224": s224["orig_sizes"], "os448": s448["orig_sizes"],
        }
    print("Data loaded.", flush=True)

    all_rows = []
    n_params = sum(p.numel() for p in PatchDecoder(in_dim=384).parameters())

    for seed in SEEDS:
        ts_seed = time.time()
        print(f"\n=== Seed {seed} ===", flush=True)

        # Train 224 decoder
        m224, vd224 = train_decoder(
            kvasir_224["patch_tokens"][train_idx_224], masks_224_tr,
            kvasir_224["patch_tokens"][val_idx_224], masks_224_va,
            GRID_224, bs=32, seed=seed)

        # Train 448 decoder (primary)
        m448, vd448 = train_decoder(
            kvasir_448["patch_tokens"][train_idx_448], masks_448_tr,
            kvasir_448["patch_tokens"][val_idx_448], masks_448_va,
            GRID_448, bs=16, seed=seed)

        # Train second 448 decoder (different seed) for variant 4
        seed_alt = seed + 100
        m448b, vd448b = train_decoder(
            kvasir_448["patch_tokens"][train_idx_448], masks_448_tr,
            kvasir_448["patch_tokens"][val_idx_448], masks_448_va,
            GRID_448, bs=16, seed=seed_alt)

        print(f"  224 val={vd224:.4f}  448 val={vd448:.4f}  448b val={vd448b:.4f}", flush=True)

        for sn in sites:
            sd = site_data[sn]
            tok224, tok448 = sd["tok224"], sd["tok448"]
            mp448, os448 = sd["mp448"], sd["os448"]

            # Get logits/probs
            logits_224 = get_logits(m224, tok224, GRID_224)
            probs_224 = sigmoid(logits_224)  # [N, 128, 128]
            logits_448 = get_logits(m448, tok448, GRID_448)
            probs_448 = sigmoid(logits_448)  # [N, 256, 256]
            logits_448b = get_logits(m448b, tok448, GRID_448)
            probs_448b = sigmoid(logits_448b)

            # ── Variant 0: source-only 448 ──
            metrics_v0 = aggregate(eval_at_orig(probs_448, mp448, os448, MASK_448))

            # ── Variant 1: ensemble 224+448 (average probs, no fine-tuning) ──
            # Upsample 224 probs to 256x256, average with 448 probs
            probs_224_up = np.stack([
                np.array(Image.fromarray(p.astype(np.float32), mode="F")
                         .resize((MASK_448, MASK_448), Image.BILINEAR))
                for p in probs_224
            ])
            probs_ens = (probs_224_up + probs_448) / 2.0
            metrics_v1 = aggregate(eval_at_orig(probs_ens, mp448, os448, MASK_448))

            # ── Variant 2: self-training on 448's own pseudo-labels ──
            preds_448_bin = (probs_448 >= 0.5).astype(np.float32)
            t_ft = time.time()
            m_v2 = finetune_uniform(m448, tok448, preds_448_bin, GRID_448, seed=seed)
            t_v2 = time.time() - t_ft
            probs_v2 = sigmoid(get_logits(m_v2, tok448, GRID_448))
            metrics_v2 = aggregate(eval_at_orig(probs_v2, mp448, os448, MASK_448))

            # ── Variant 3: ours (224 vs 448 agreement weights) ──
            preds_224_bin_up = (probs_224_up >= 0.5).astype(np.float32)
            agreement_34 = (preds_224_bin_up == preds_448_bin).astype(np.float32)
            weights_v3 = np.where(agreement_34 > 0.5, 1.0, 0.1).astype(np.float32)
            t_ft = time.time()
            m_v3 = finetune_weighted(m448, tok448, preds_448_bin, weights_v3, GRID_448, seed=seed)
            t_v3 = time.time() - t_ft
            probs_v3 = sigmoid(get_logits(m_v3, tok448, GRID_448))
            metrics_v3 = aggregate(eval_at_orig(probs_v3, mp448, os448, MASK_448))

            # ── Variant 4: agreement between two 448 decoders ──
            preds_448b_bin = (probs_448b >= 0.5).astype(np.float32)
            agreement_4 = (preds_448_bin == preds_448b_bin).astype(np.float32)
            weights_v4 = np.where(agreement_4 > 0.5, 1.0, 0.1).astype(np.float32)
            t_ft = time.time()
            m_v4 = finetune_weighted(m448, tok448, preds_448_bin, weights_v4, GRID_448, seed=seed)
            t_v4 = time.time() - t_ft
            probs_v4 = sigmoid(get_logits(m_v4, tok448, GRID_448))
            metrics_v4 = aggregate(eval_at_orig(probs_v4, mp448, os448, MASK_448))

            agree_224_448 = float(agreement_34.mean())
            agree_448_448 = float(agreement_4.mean())

            for vid, vname, met, adapt_time in [
                (0, "source_448", metrics_v0, 0.0),
                (1, "ensemble_224_448", metrics_v1, 0.0),
                (2, "self_train_uniform", metrics_v2, t_v2),
                (3, "ours_agree_224_448", metrics_v3, t_v3),
                (4, "agree_448_448", metrics_v4, t_v4),
            ]:
                all_rows.append({
                    "seed": seed, "site": sn, "variant_id": vid,
                    "variant": vname,
                    "dice_pos": met["dice_pos"], "dice_all": met["dice_all"],
                    "dice_pooled": met["dice_pooled"], "fp_rate": met["fp_rate"],
                    "n_pos": met["n_pos"], "n_neg": met["n_neg"],
                    "adapt_time_s": adapt_time,
                    "agree_224_448": agree_224_448,
                    "agree_448_448": agree_448_448,
                })

            print(f"  {sn}: v0={metrics_v0['dice_pos']:.4f} v1={metrics_v1['dice_pos']:.4f} "
                  f"v2={metrics_v2['dice_pos']:.4f} v3={metrics_v3['dice_pos']:.4f} "
                  f"v4={metrics_v4['dice_pos']:.4f}", flush=True)

        print(f"  Seed time: {time.time() - ts_seed:.0f}s", flush=True)

    # Save raw data
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "ablation_raw.csv", index=False)

    # Print aggregated tables
    print(f"\n{'='*140}")
    print("TABLE: ABLATION — POSITIVES-ONLY DICE (mean ± std over 5 seeds, original resolution)")
    print(f"{'='*140}")

    variants = ["source_448", "ensemble_224_448", "self_train_uniform",
                "ours_agree_224_448", "agree_448_448"]
    header = f"{'site':>15}"
    for v in variants:
        header += f" {v:>24}"
    print(header)
    print("-" * 140)

    for sn in sites:
        line = f"{sn:>15}"
        for v in variants:
            sub = df[(df.site == sn) & (df.variant == v)]
            m = sub.dice_pos.mean()
            s = sub.dice_pos.std()
            line += f" {m:>10.4f}±{s:<10.4f}"
        print(line)

    # Delta table
    print(f"\n{'='*120}")
    print("TABLE: ABLATION — DELTA vs SOURCE-ONLY 448")
    print(f"{'='*120}")
    header2 = f"{'site':>15}"
    for v in variants[1:]:
        header2 += f" {v:>24}"
    print(header2)
    print("-" * 120)

    for sn in sites:
        base = df[(df.site == sn) & (df.variant == "source_448")]
        line = f"{sn:>15}"
        for v in variants[1:]:
            sub = df[(df.site == sn) & (df.variant == v)]
            deltas = []
            for seed in SEEDS:
                b = base[base.seed == seed].dice_pos.values[0]
                s = sub[sub.seed == seed].dice_pos.values[0]
                deltas.append(s - b)
            line += f" {np.mean(deltas):>+10.4f}±{np.std(deltas):<10.4f}"
        print(line)

    # All-images Dice + pooled Dice + FP rate
    print(f"\n{'='*140}")
    print("TABLE: ALL-IMAGES DICE / POOLED DICE / FP RATE (mean over 5 seeds)")
    print(f"{'='*140}")
    print(f"{'site':>15} {'variant':>24} {'dice_all':>10} {'dice_pooled':>12} {'fp_rate':>10}")
    print("-" * 80)
    for sn in sites:
        for v in variants:
            sub = df[(df.site == sn) & (df.variant == v)]
            print(f"{sn:>15} {v:>24} {sub.dice_all.mean():>10.4f} "
                  f"{sub.dice_pooled.mean():>12.4f} {sub.fp_rate.mean():>10.3f}")

    # Adaptation time + params
    print(f"\n{'='*80}")
    print("TABLE: COST")
    print(f"{'='*80}")
    print(f"{'variant':>24} {'adapt_time_s':>14} {'extra_params':>14}")
    print("-" * 60)
    for v in variants:
        sub = df[df.variant == v]
        t = sub.adapt_time_s.mean()
        extra = n_params if v in ("self_train_uniform", "ours_agree_224_448", "agree_448_448") else 0
        print(f"{v:>24} {t:>14.1f} {extra:>14,}")

    # Win counts
    print(f"\n{'='*60}")
    print("WINS (positive mean delta on how many sites)")
    print(f"{'='*60}")
    for v in variants[1:]:
        wins = 0
        for sn in sites:
            base = df[(df.site == sn) & (df.variant == "source_448")].dice_pos.mean()
            val = df[(df.site == sn) & (df.variant == v)].dice_pos.mean()
            if val > base:
                wins += 1
        print(f"  {v}: {wins}/6")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
