"""Control 1: Resolution comparison.

Recompute Phase 2 alignment results at ORIGINAL mask resolution
(bilinear upsample of 128x128 probs to original size, threshold 0.5).
Compare with 128x128 results and flag differences > 0.01.
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
from eval import (GRID, N_PATCHES, bootstrap_ci, dice_score, load_gt_mask,
                  load_site_features, sigmoid, upsample_probs)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

TARGET_SITES = [f"polypgen_C{i}" for i in range(1, 7)]


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


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
def decoder_logits_128(model, tokens):
    model.eval()
    out = []
    for i in range(0, len(tokens), 64):
        batch = tokens[i:i + 64]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_both_res(model, tokens, mask_paths, orig_sizes, gt_128):
    logits = decoder_logits_128(model, tokens)
    probs_128 = sigmoid(logits)
    rows_128, rows_orig = [], []
    for i in range(len(tokens)):
        p128 = probs_128[i]
        pred128 = (p128 >= 0.5).astype(np.uint8)
        g128 = (gt_128[i] >= 0.5).astype(np.uint8)
        is_empty_128 = g128.sum() == 0
        rows_128.append({
            "dice": dice_score(pred128, g128),
            "is_empty": is_empty_128,
        })

        ow, oh = int(orig_sizes[i][0]), int(orig_sizes[i][1])
        prob_up = np.array(Image.fromarray(np.asarray(p128, dtype=np.float32), mode="F")
                           .resize((ow, oh), Image.BILINEAR))
        pred_orig = (prob_up >= 0.5).astype(np.uint8)
        gt_orig = load_gt_mask(str(mask_paths[i]))
        is_empty_orig = gt_orig.sum() == 0
        rows_orig.append({
            "dice": dice_score(pred_orig, gt_orig),
            "is_empty": is_empty_orig,
        })
    return rows_128, rows_orig


def summarize(res_list):
    dices = np.array([r["dice"] for r in res_list])
    pos_dices = np.array([r["dice"] for r in res_list if not r["is_empty"]])
    return {
        "dice": float(dices.mean()),
        "dice_pos": float(pos_dices.mean()) if len(pos_dices) > 0 else float("nan"),
    }


def channel_standardize(X, mu_s, sigma_s, mu_t, sigma_t, eps=1e-6):
    return ((X - mu_t) / (sigma_t + eps) * sigma_s + mu_s).astype(np.float32)


def coral_stats(X):
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), np.cov(X, rowvar=False)


def coral_transform(X_target, mu_s, C_s, mu_t, C_t, reg=1e-5):
    D = C_s.shape[0]
    I = np.eye(D)
    d_t, V_t = np.linalg.eigh(C_t + reg * I)
    d_t = np.maximum(d_t, reg)
    Ct_isqrt = V_t @ np.diag(1.0 / np.sqrt(d_t)) @ V_t.T
    d_s, V_s = np.linalg.eigh(C_s + reg * I)
    d_s = np.maximum(d_s, reg)
    Cs_sqrt = V_s @ np.diag(np.sqrt(d_s)) @ V_s.T
    W = Ct_isqrt @ Cs_sqrt
    X = np.asarray(X_target, dtype=np.float64)
    return ((X - mu_t) @ W + mu_s).astype(np.float32)


@torch.no_grad()
def adapt_bn(model, target_tokens):
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    targets = {nm: load_site_features(FEAT_DIR, nm) for nm in TARGET_SITES}

    print("Caching masks at 128x128 ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    target_m128 = {nm: load_masks_128(targets[nm]["mask_paths"]) for nm in TARGET_SITES}

    print("Training decoder (seed 42) ...", flush=True)
    ts = time.time()
    model, vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)

    # Source stats
    X_src = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    mu_s = X_src.mean(axis=0)
    sigma_s = X_src.std(axis=0)
    mu_s_cor, C_s_cor = coral_stats(X_src)
    lam = 1e-3 * np.diag(C_s_cor).mean()

    sites = [("kvasir_val", kvasir, val_idx)]
    for nm in TARGET_SITES:
        sites.append((nm, targets[nm], np.arange(len(targets[nm]["filenames"]))))

    rows_all = []
    for site_name, sd, indices in sites:
        n = len(indices)
        print(f"\n  {site_name} ({n}) ...", flush=True)
        tokens = sd["patch_tokens"][indices]
        mask_paths = sd["mask_paths"][indices]
        orig_sizes = sd["orig_sizes"][indices]
        gt_128 = kvasir_m128[indices] if site_name == "kvasir_val" else target_m128[site_name]

        X_t = tokens.reshape(-1, 384).astype(np.float64)
        mu_t = X_t.mean(axis=0)
        sigma_t = X_t.std(axis=0)
        mu_t_cor, C_t_cor = coral_stats(X_t)

        for variant in ["A_none", "B_channel_std", "C_coral", "D_adabn"]:
            if variant == "A_none":
                tok, mdl = tokens, model
            elif variant == "B_channel_std":
                tok = channel_standardize(
                    tokens.reshape(-1, 384), mu_s, sigma_s, mu_t, sigma_t
                ).reshape(n, N_PATCHES, 384)
                mdl = model
            elif variant == "C_coral":
                tok = coral_transform(X_t, mu_s_cor, C_s_cor, mu_t_cor, C_t_cor, reg=lam
                ).reshape(n, N_PATCHES, 384)
                mdl = model
            elif variant == "D_adabn":
                tok, mdl = tokens, adapt_bn(model, tokens)

            r128, rorig = eval_both_res(mdl, tok, mask_paths, orig_sizes, gt_128)
            s128 = summarize(r128)
            sorig = summarize(rorig)

            d128 = np.array([r["dice"] for r in r128])
            dorig = np.array([r["dice"] for r in rorig])
            diff = dorig - d128
            ci = bootstrap_ci(diff)

            rows_all.append({
                "variant": variant, "site": site_name,
                "dice_128": s128["dice"], "dice_pos_128": s128["dice_pos"],
                "dice_orig": sorig["dice"], "dice_pos_orig": sorig["dice_pos"],
                "diff_dice": sorig["dice"] - s128["dice"],
                "diff_dice_pos": sorig["dice_pos"] - s128["dice_pos"],
            })

    df = pd.DataFrame(rows_all)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 140)
    print("TABLE: RESOLUTION COMPARISON (128x128 vs original, decoder seed 42)")
    print("All Dice at threshold=0.5. 128x128 = native decoder output. orig = bilinear upsample to original mask size.")
    print("=" * 140)
    print(df.to_string(index=False))

    flagged = df[df["diff_dice_pos"].abs() > 0.01]
    print(f"\n  Differences > 0.01 in dice_pos: {len(flagged)}")
    if len(flagged) > 0:
        print(flagged[["variant", "site", "dice_pos_128", "dice_pos_orig", "diff_dice_pos"]].to_string(index=False))

    df.to_csv(OUT_DIR / "ctrl_resolution.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
