"""Control 3: FOV-restricted alignment statistics.

Recompute target mean/std/covariance using only patches inside the
visible field of view (>=50% FOV coverage per patch). Exclude C4's
teal info-panel region if detectable. Apply correction to all patches.
Rerun channel-std, CORAL, AdaBN with bootstrap CIs.
"""

import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import GRID, N_PATCHES, bootstrap_ci, dice_score, load_site_features, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
FOV_THRESH = 12

TARGET_SITES = [f"polypgen_C{i}" for i in range(1, 7)]


def estimate_fov_mask(img_path, target_size=224, patch_size=14):
    """Return per-patch FOV fraction (16x16) and optional teal mask."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.ones((GRID, GRID), dtype=np.float32), False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, FOV_THRESH, 255, cv2.THRESH_BINARY)
    n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        fov = np.ones(gray.shape, dtype=np.uint8) * 255
    else:
        areas = comp_stats[1:, cv2.CC_STAT_AREA]
        largest = np.argmax(areas) + 1
        fov = np.zeros(gray.shape, dtype=np.uint8)
        fov[labels == largest] = 255
        contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fov, contours, -1, 255, cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)

    # Detect C4 teal panel: HSV check for teal/cyan in top-left region
    has_teal = False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    teal_mask = ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 100) &
                 (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))
    h, w = img.shape[:2]
    teal_corner = teal_mask[:h // 4, :w // 3]
    if teal_corner.mean() > 0.3:
        has_teal = True
        teal_full = teal_mask.astype(np.uint8) * 255
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        teal_full = cv2.dilate(teal_full, kernel2)
        fov = cv2.bitwise_and(fov, cv2.bitwise_not(teal_full))

    # Resize FOV to 224x224 and compute per-patch coverage
    fov_224 = cv2.resize(fov, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    grid = target_size // patch_size
    patch_fov = np.zeros((grid, grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            patch = fov_224[r * patch_size:(r + 1) * patch_size,
                           c * patch_size:(c + 1) * patch_size]
            patch_fov[r, c] = (patch > 0).mean()

    return patch_fov, has_teal


def get_image_paths(site_name):
    """Resolve image paths for a site."""
    import yaml
    with open(ROOT / "configs" / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][site_name]
    img_dir = ROOT / "data" / "raw" / ds["images"]
    return img_dir


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


def eval_128(model, tokens, gt_128):
    logits = decoder_logits_128(model, tokens)
    probs = sigmoid(logits)
    rows = []
    for i in range(len(tokens)):
        pred = (probs[i] >= 0.5).astype(np.uint8)
        gt = (gt_128[i] >= 0.5).astype(np.uint8)
        is_empty = gt.sum() == 0
        rows.append({
            "dice": dice_score(pred, gt),
            "is_empty": is_empty,
        })
    return rows


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

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    target_m128 = {nm: load_masks_128(targets[nm]["mask_paths"]) for nm in TARGET_SITES}

    print("Training decoder ...", flush=True)
    ts = time.time()
    model, vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)

    # Source statistics (all patches — Kvasir has full FOV)
    X_src = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    mu_s = X_src.mean(axis=0)
    sigma_s = X_src.std(axis=0)
    mu_s_cor, C_s_cor = coral_stats(X_src)
    lam = 1e-3 * np.diag(C_s_cor).mean()

    # For each site, compute FOV masks and FOV-restricted target stats
    all_rows = []
    for site_name in TARGET_SITES:
        sd = targets[site_name]
        n = len(sd["filenames"])
        gt_128 = target_m128[site_name]
        tokens = sd["patch_tokens"]
        print(f"\n  {site_name} ({n}) ...", flush=True)

        # Compute FOV mask per image
        img_dir = get_image_paths(site_name)
        fov_patches = []
        n_teal = 0
        for i in range(n):
            fn = sd["filenames"][i]
            img_files = list(img_dir.glob(f"{fn}.*"))
            if img_files:
                pfov, has_teal = estimate_fov_mask(str(img_files[0]))
            else:
                pfov, has_teal = np.ones((GRID, GRID), dtype=np.float32), False
            fov_patches.append(pfov.ravel())
            if has_teal:
                n_teal += 1
        fov_patches = np.stack(fov_patches)
        fov_mask = fov_patches >= 0.5
        n_fov = fov_mask.sum()
        n_total = fov_mask.size
        print(f"    FOV patches: {n_fov}/{n_total} ({n_fov / n_total:.1%}), teal panels: {n_teal}", flush=True)

        # FOV-restricted target stats
        all_tok_flat = tokens.reshape(-1, 384).astype(np.float64)
        fov_flat = fov_mask.ravel()
        X_fov = all_tok_flat[fov_flat]
        mu_t_fov = X_fov.mean(axis=0)
        sigma_t_fov = X_fov.std(axis=0)
        mu_t_fov_cor, C_t_fov_cor = coral_stats(X_fov)

        # Unrestricted target stats (for comparison)
        mu_t_all = all_tok_flat.mean(axis=0)
        sigma_t_all = all_tok_flat.std(axis=0)
        mu_t_all_cor, C_t_all_cor = coral_stats(all_tok_flat)

        print(f"    mu diff (FOV vs all): {np.abs(mu_t_fov - mu_t_all).mean():.4f}", flush=True)

        for stat_label, mu_t, sigma_t, mu_tc, C_tc in [
            ("all_patches", mu_t_all, sigma_t_all, mu_t_all_cor, C_t_all_cor),
            ("fov_patches", mu_t_fov, sigma_t_fov, mu_t_fov_cor, C_t_fov_cor),
        ]:
            # Baseline
            res_A = eval_128(model, tokens, gt_128)
            dices_A = np.array([r["dice"] for r in res_A])
            pos_mask = np.array([not r["is_empty"] for r in res_A])

            for variant, get_tok_mdl in [
                ("A_none", lambda: (tokens, model)),
                ("B_channel_std", lambda: (channel_standardize(
                    all_tok_flat, mu_s, sigma_s, mu_t, sigma_t
                ).reshape(n, N_PATCHES, 384), model)),
                ("C_coral", lambda: (coral_transform(
                    all_tok_flat, mu_s_cor, C_s_cor, mu_tc, C_tc, reg=lam
                ).reshape(n, N_PATCHES, 384), model)),
                ("D_adabn", lambda: (tokens, adapt_bn(model, tokens))),
            ]:
                tok, mdl = get_tok_mdl()
                res = eval_128(mdl, tok, gt_128)
                dices_v = np.array([r["dice"] for r in res])
                d_all = float(dices_v.mean())
                d_pos = float(dices_v[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan")

                row = {
                    "site": site_name, "stat_source": stat_label,
                    "variant": variant,
                    "dice": d_all, "dice_pos": d_pos,
                    "n_fov_patches": int(n_fov), "n_total_patches": int(n_total),
                    "n_teal": n_teal,
                }
                if variant != "A_none":
                    delta = dices_v - dices_A
                    ci = bootstrap_ci(delta)
                    row["delta"] = float(delta.mean())
                    row["CI_lo"] = ci[0]
                    row["CI_hi"] = ci[1]
                    if pos_mask.sum() > 0:
                        dp = dices_v[pos_mask] - dices_A[pos_mask]
                        ci_p = bootstrap_ci(dp)
                        row["delta_pos"] = float(dp.mean())
                        row["CI_pos_lo"] = ci_p[0]
                        row["CI_pos_hi"] = ci_p[1]

                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 160)
    print("TABLE: FOV-RESTRICTED vs ALL-PATCHES ALIGNMENT (decoder, seed 42, 128x128 eval)")
    print("=" * 160)
    cols = ["site", "stat_source", "variant", "dice", "dice_pos"]
    if "delta" in df.columns:
        cols += ["delta", "CI_lo", "CI_hi"]
    print(df[cols].to_string(index=False))

    df.to_csv(OUT_DIR / "ctrl_fov.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
