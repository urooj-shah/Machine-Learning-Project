"""Phase 3: Prior-matched source statistics for alignment.

Tests whether re-mixing source class-conditional statistics at the
estimated target prior improves alignment. Decoder only, seed 42.
"""

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
from eval import (
    GRID, N_PATCHES, bootstrap_ci, dice_score,
    load_site_features, saerens_em, sigmoid,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "06_poc"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        x = np.ascontiguousarray(x)
        y = self.masks[idx]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


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


def train_decoder_fn(train_tokens, train_masks, val_tokens, val_masks,
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
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
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


def eval_128(model, tokens, gt_128, threshold=0.5):
    logits = decoder_logits_128(model, tokens)
    probs = sigmoid(logits)
    rows = []
    for i in range(len(tokens)):
        pred = (probs[i] >= threshold).astype(np.uint8)
        gt = (gt_128[i] >= 0.5).astype(np.uint8)
        pb, gb = pred.astype(bool), gt.astype(bool)
        tp = int((pb & gb).sum())
        fp = int((pb & ~gb).sum())
        fn = int((~pb & gb).sum())
        is_empty = (gb.sum() == 0)
        rows.append({
            "dice": dice_score(pred, gt),
            "is_empty": is_empty,
            "fp_on_empty": bool(is_empty and pb.sum() > 0),
            "tp": tp, "fp": fp, "fn": fn,
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


def remix_channel_stats(mu_fg, sigma_fg, mu_bg, sigma_bg, pi_t):
    mu = pi_t * mu_fg + (1 - pi_t) * mu_bg
    var = (pi_t * (sigma_fg**2 + mu_fg**2) +
           (1 - pi_t) * (sigma_bg**2 + mu_bg**2) - mu**2)
    return mu, np.sqrt(np.maximum(var, 1e-10))


def remix_covariance(mu_fg, C_fg, mu_bg, C_bg, pi_t):
    mu = pi_t * mu_fg + (1 - pi_t) * mu_bg
    C = (pi_t * (C_fg + np.outer(mu_fg, mu_fg)) +
         (1 - pi_t) * (C_bg + np.outer(mu_bg, mu_bg)) -
         np.outer(mu, mu))
    return mu, C


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

    targets = {}
    for name in TARGET_SITES:
        targets[name] = load_site_features(FEAT_DIR, name)

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    target_m128 = {}
    for name in TARGET_SITES:
        target_m128[name] = load_masks_128(targets[name]["mask_paths"])

    # Train decoder
    print("Training decoder (seed 42) ...", flush=True)
    decoder, vd = train_decoder_fn(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    pi_src_dec = float(kvasir_m128[train_idx].mean())
    print(f"  Val Dice: {vd:.4f}, pi_source: {pi_src_dec:.4f}")

    # Source class-conditional statistics
    print("Computing class-conditional statistics ...", flush=True)
    X_src = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    y_src = kvasir["patch_fg_binary"][train_idx].reshape(-1)

    fg_mask = (y_src == 1)
    bg_mask = (y_src == 0)
    pi_src_patch = float(fg_mask.mean())
    print(f"  Source FG rate: {pi_src_patch:.4f} ({fg_mask.sum()} FG, {bg_mask.sum()} BG)")

    X_fg, X_bg = X_src[fg_mask], X_src[bg_mask]
    mu_fg_ch, sigma_fg_ch = X_fg.mean(axis=0), X_fg.std(axis=0)
    mu_bg_ch, sigma_bg_ch = X_bg.mean(axis=0), X_bg.std(axis=0)
    mu_fg_cor, C_fg_cor = coral_stats(X_fg)
    mu_bg_cor, C_bg_cor = coral_stats(X_bg)

    mu_s_marginal = X_src.mean(axis=0)
    sigma_s_marginal = X_src.std(axis=0)
    mu_s_coral, C_s_coral = coral_stats(X_src)
    lam = 1e-3 * np.diag(C_s_coral).mean()

    # Evaluate
    sites = [("kvasir_val", kvasir, val_idx, kvasir_m128[val_idx])]
    for nm in TARGET_SITES:
        n_t = len(targets[nm]["filenames"])
        sites.append((nm, targets[nm], np.arange(n_t), target_m128[nm]))

    variants = [
        "A_none", "B_channel_std", "B_matched_em", "B_matched_oracle",
        "C_coral", "C_matched_em", "C_matched_oracle",
    ]
    all_rows = []

    for site_name, sd, indices, gt_128 in sites:
        n = len(indices)
        print(f"\n  {site_name} ({n}) ...", flush=True)

        tokens_orig = sd["patch_tokens"][indices]
        X_tgt = tokens_orig.reshape(-1, 384).astype(np.float64)
        mu_t_ch = X_tgt.mean(axis=0)
        sigma_t_ch = X_tgt.std(axis=0)
        mu_t_cor, C_t_cor = coral_stats(X_tgt)

        pi_true_patch = float(sd["patch_fg_binary"][indices].reshape(-1).mean())
        pi_true_pixel = float(gt_128.mean())

        dec_logits = decoder_logits_128(decoder, tokens_orig)
        dec_probs = sigmoid(dec_logits).reshape(-1).astype(np.float64)
        pi_em = saerens_em(dec_probs, pi_source=pi_src_dec)
        print(f"    pi_true_patch={pi_true_patch:.4f}, pi_em={pi_em:.4f}")

        mu_em_ch, sigma_em_ch = remix_channel_stats(mu_fg_ch, sigma_fg_ch, mu_bg_ch, sigma_bg_ch, pi_em)
        mu_or_ch, sigma_or_ch = remix_channel_stats(mu_fg_ch, sigma_fg_ch, mu_bg_ch, sigma_bg_ch, pi_true_patch)
        mu_em_cor, C_em_cor = remix_covariance(mu_fg_cor, C_fg_cor, mu_bg_cor, C_bg_cor, pi_em)
        mu_or_cor, C_or_cor = remix_covariance(mu_fg_cor, C_fg_cor, mu_bg_cor, C_bg_cor, pi_true_patch)

        res_A = eval_128(decoder, tokens_orig, gt_128)
        dices_A = np.array([r["dice"] for r in res_A])
        pos_mask = np.array([not r["is_empty"] for r in res_A])

        variant_tokens = {
            "A_none": tokens_orig,
            "B_channel_std": channel_standardize(
                tokens_orig.reshape(-1, 384), mu_s_marginal, sigma_s_marginal,
                mu_t_ch, sigma_t_ch).reshape(n, N_PATCHES, 384),
            "B_matched_em": channel_standardize(
                tokens_orig.reshape(-1, 384), mu_em_ch, sigma_em_ch,
                mu_t_ch, sigma_t_ch).reshape(n, N_PATCHES, 384),
            "B_matched_oracle": channel_standardize(
                tokens_orig.reshape(-1, 384), mu_or_ch, sigma_or_ch,
                mu_t_ch, sigma_t_ch).reshape(n, N_PATCHES, 384),
            "C_coral": coral_transform(
                X_tgt, mu_s_coral, C_s_coral, mu_t_cor, C_t_cor, reg=lam
            ).reshape(n, N_PATCHES, 384),
            "C_matched_em": coral_transform(
                X_tgt, mu_em_cor, C_em_cor, mu_t_cor, C_t_cor, reg=lam
            ).reshape(n, N_PATCHES, 384),
            "C_matched_oracle": coral_transform(
                X_tgt, mu_or_cor, C_or_cor, mu_t_cor, C_t_cor, reg=lam
            ).reshape(n, N_PATCHES, 384),
        }

        for variant in variants:
            tok = variant_tokens[variant]
            res = eval_128(decoder, tok, gt_128)
            dices_v = np.array([r["dice"] for r in res])
            m = {
                "dice": float(dices_v.mean()),
                "dice_pos": float(dices_v[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan"),
                "fp_rate_empty": float(np.mean([r["fp_on_empty"] for r in res if r["is_empty"]])) if (~pos_mask).sum() > 0 else 0.0,
            }

            deltas = dices_v - dices_A
            ci = bootstrap_ci(deltas) if variant != "A_none" else (0.0, 0.0)
            dp = dices_v[pos_mask] - dices_A[pos_mask] if pos_mask.sum() > 0 and variant != "A_none" else np.zeros(1)
            ci_p = bootstrap_ci(dp) if variant != "A_none" and pos_mask.sum() > 0 else (0.0, 0.0)

            all_rows.append({
                "variant": variant, "site": site_name, "n": n,
                "dice": m["dice"], "dice_pos": m["dice_pos"],
                "fp_rate_empty": m["fp_rate_empty"],
                "delta": float(deltas.mean()),
                "CI_lo": ci[0], "CI_hi": ci[1],
                "delta_pos": float(dp.mean()),
                "CI_pos_lo": ci_p[0], "CI_pos_hi": ci_p[1],
                "pi_true": pi_true_patch, "pi_em": pi_em,
            })

    # Print
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    df = pd.DataFrame(all_rows)
    cols = ["variant", "site", "dice", "dice_pos", "fp_rate_empty",
            "delta", "CI_lo", "CI_hi", "delta_pos", "CI_pos_lo", "CI_pos_hi"]
    print("\n" + "=" * 160)
    print("TABLE: PRIOR-MATCHED ALIGNMENT (decoder, seed 42, 128x128 eval)")
    print("=" * 160)
    print(df[cols].to_string(index=False))

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    for variant in variants[1:]:
        wins = 0
        kv_drop = 0.0
        for _, row in df.iterrows():
            if row["variant"] == variant:
                if row["site"] == "kvasir_val":
                    base_kv = df[(df["variant"] == "A_none") & (df["site"] == "kvasir_val")]
                    kv_drop = base_kv.iloc[0]["dice_pos"] - row["dice_pos"]
                elif row["site"].startswith("polypgen"):
                    if row["CI_pos_lo"] > 0:
                        wins += 1
        meets = wins >= 3 and kv_drop <= 0.01
        print(f"  {variant}: {wins}/6 wins, Kv drop={kv_drop:.4f}, {'MEETS' if meets else 'NO'}")

    df.to_csv(OUT_DIR / "phase3_prior_matched.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
