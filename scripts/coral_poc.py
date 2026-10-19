"""CORAL + EM proof of concept: feature alignment + prior correction.

Tests whether CORAL (covariate shift fix) combined with Saerens EM
(label shift fix) improves cross-hospital polyp segmentation Dice.
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
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import (
    GRID, N_PATCHES, bootstrap_ci, dice_score, load_gt_mask,
    load_site_features, logit_shift, saerens_em, sigmoid, upsample_probs,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "06_coral_poc"
SEED = 42
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


# ── CORAL functions ───────────────────────────────────────────────

def coral_stats(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    return mu, C


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
    return (X - mu_t) @ W + mu_s


# ── Reused from audit_experiment.py ───────────────────────────────

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


def train_decoder(train_tokens, train_masks, val_tokens=None, val_masks=None,
                  n_epochs=80, lr=1e-3, seed=SEED):
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
        if val_tokens is not None:
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
                    td += ((2*inter+1)/(union+1)).sum().item()
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
    for i in range(0, len(tokens), 32):
        batch = tokens[i:i+32]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def up_lr(prob_256, ow, oh):
    g = np.asarray(prob_256, dtype=np.float32).reshape(GRID, GRID)
    return upsample_probs(g, ow, oh)


def up_dec(prob_128, ow, oh):
    a = np.asarray(prob_128, dtype=np.float32)
    img = Image.fromarray(a, mode="F")
    return np.array(img.resize((int(ow), int(oh)), Image.BILINEAR))


def eval_maps(pred_maps, gt_masks, threshold=0.5):
    rows = []
    for pm, gt in zip(pred_maps, gt_masks):
        pred = (pm >= threshold).astype(np.uint8)
        pb, gb = pred.astype(bool), gt.astype(bool)
        tp = int((pb & gb).sum())
        fp = int((pb & ~gb).sum())
        fn = int((~pb & gb).sum())
        is_empty = (gb.sum() == 0)
        rows.append({
            "dice": dice_score(pred, gt),
            "prec": float(tp / max(tp+fp, 1)),
            "rec": float(tp / max(tp+fn, 1)),
            "is_empty": is_empty,
            "fp_on_empty": bool(is_empty and pb.sum() > 0),
        })
    return rows


# ── Evaluation helper ─────────────────────────────────────────────

def evaluate_condition(lr_probs_256, lr_logits_256, dec_logits_128,
                       site_data, indices, gt_masks):
    n = len(indices)
    lr_p_img = [lr_probs_256[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]
    dec_p = sigmoid(dec_logits_128)

    lr_maps, dec_maps = [], []
    for i, idx in enumerate(indices):
        ow, oh = int(site_data["orig_sizes"][idx][0]), int(site_data["orig_sizes"][idx][1])
        lr_maps.append(up_lr(lr_p_img[i], ow, oh))
        dec_maps.append(up_dec(dec_p[i], ow, oh))

    lr_res = eval_maps(lr_maps, gt_masks)
    dec_res = eval_maps(dec_maps, gt_masks)
    return lr_res, dec_res


def summarize(res_list):
    dices = np.array([r["dice"] for r in res_list])
    empty = [r for r in res_list if r["is_empty"]]
    fp_e = [r["fp_on_empty"] for r in empty] if empty else []
    return {
        "dice": float(dices.mean()),
        "n_empty": len(empty),
        "fp_rate_empty": float(np.mean(fp_e)) if fp_e else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
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

    # Train LR
    print("Training LR ...", flush=True)
    clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                             class_weight="balanced")
    X_train = kvasir["patch_tokens"][train_idx].reshape(-1, 384)
    y_train = kvasir["patch_fg_binary"][train_idx].reshape(-1)
    clf.fit(X_train, y_train)
    pi_source_lr = 0.5

    # Train decoder
    print("Training decoder ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    pi_source_dec = float(kvasir_m128[train_idx].mean())
    decoder, dec_vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  Decoder val Dice={dec_vd:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # Compute source CORAL stats
    print("Computing source CORAL stats ...", flush=True)
    X_src_64 = X_train.astype(np.float64)
    mu_s, C_s = coral_stats(X_src_64)

    # Sanity check: CORAL(source → source) ≈ identity
    X_src_aligned = coral_transform(X_src_64[:1000], mu_s, C_s, mu_s, C_s)
    err = np.abs(X_src_aligned - X_src_64[:1000]).max()
    print(f"  CORAL self-alignment max error: {err:.2e} (should be ~1e-10)")

    # Evaluate all sites
    sites = [("kvasir_val", kvasir, val_idx)]
    for nm in TARGET_SITES:
        sites.append((nm, targets[nm], np.arange(len(targets[nm]["filenames"]))))

    conditions = ["none", "EM", "CORAL", "CORAL+EM"]
    all_lr, all_dec = [], []

    for site_name, sd, indices in sites:
        n = len(indices)
        print(f"\n  Evaluating {site_name} ({n}) ...", flush=True)

        gt_masks = [load_gt_mask(str(sd["mask_paths"][idx])) for idx in indices]
        m128 = load_masks_128(sd["mask_paths"][indices])

        # Original features
        X_orig = sd["patch_tokens"][indices].reshape(-1, 384)
        lr_l_orig = clf.decision_function(X_orig).astype(np.float64)
        lr_p_orig = clf.predict_proba(X_orig)[:, 1].astype(np.float32)
        tokens_orig = sd["patch_tokens"][indices]
        dl_orig = decoder_logits_128(decoder, tokens_orig)

        # CORAL-aligned features
        mu_t, C_t = coral_stats(X_orig.astype(np.float64))
        X_aligned_64 = coral_transform(X_orig.astype(np.float64), mu_s, C_s, mu_t, C_t)
        X_aligned = X_aligned_64.astype(np.float32)

        lr_l_coral = clf.decision_function(X_aligned).astype(np.float64)
        lr_p_coral = clf.predict_proba(X_aligned)[:, 1].astype(np.float32)
        tokens_coral = X_aligned.reshape(n, N_PATCHES, 384)
        dl_coral = decoder_logits_128(decoder, tokens_coral)

        # EM estimates
        pi_em_orig_lr = saerens_em(lr_p_orig.astype(np.float64), pi_source=pi_source_lr)
        pi_em_coral_lr = saerens_em(lr_p_coral.astype(np.float64), pi_source=pi_source_lr)

        dp_orig = sigmoid(dl_orig).reshape(-1).astype(np.float64)
        dp_coral = sigmoid(dl_coral).reshape(-1).astype(np.float64)
        pi_em_orig_dec = saerens_em(dp_orig, pi_source=pi_source_dec)
        pi_em_coral_dec = saerens_em(dp_coral, pi_source=pi_source_dec)

        # Logit shifts
        s_em_orig_lr = logit_shift(pi_source_lr, pi_em_orig_lr)
        s_em_coral_lr = logit_shift(pi_source_lr, pi_em_coral_lr)
        s_em_orig_dec = logit_shift(pi_source_dec, pi_em_orig_dec)
        s_em_coral_dec = logit_shift(pi_source_dec, pi_em_coral_dec)

        # Build corrected probs for each condition
        conds_lr = {
            "none":     (lr_p_orig, lr_l_orig),
            "EM":       (sigmoid(lr_l_orig + s_em_orig_lr).astype(np.float32), None),
            "CORAL":    (lr_p_coral, lr_l_coral),
            "CORAL+EM": (sigmoid(lr_l_coral + s_em_coral_lr).astype(np.float32), None),
        }
        conds_dec = {
            "none":     dl_orig,
            "EM":       dl_orig + s_em_orig_dec,
            "CORAL":    dl_coral,
            "CORAL+EM": dl_coral + s_em_coral_dec,
        }

        # Baseline dices for delta computation
        base_lr = None
        base_dec = None

        for cond in conditions:
            # LR
            lr_probs = conds_lr[cond][0]
            lr_p_img = [lr_probs[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]
            lr_maps = []
            for i, idx in enumerate(indices):
                ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
                lr_maps.append(up_lr(lr_p_img[i], ow, oh))
            lr_res = eval_maps(lr_maps, gt_masks)
            lr_dices = np.array([r["dice"] for r in lr_res])

            if cond == "none":
                base_lr = lr_dices
            deltas_lr = lr_dices - base_lr
            ci_lr = bootstrap_ci(deltas_lr) if cond != "none" else (0.0, 0.0)

            lr_empty = [r for r in lr_res if r["is_empty"]]
            fp_lr = float(np.mean([r["fp_on_empty"] for r in lr_empty])) if lr_empty else 0.0

            all_lr.append({
                "site": site_name, "cond": cond, "n": n,
                "dice": float(lr_dices.mean()),
                "delta": float(deltas_lr.mean()),
                "CI_lo": ci_lr[0], "CI_hi": ci_lr[1],
                "n_empty": len(lr_empty),
                "fp_rate_empty": fp_lr,
            })

            # Decoder
            dl = conds_dec[cond]
            dp = sigmoid(dl)
            dec_maps = []
            for i, idx in enumerate(indices):
                ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
                dec_maps.append(up_dec(dp[i], ow, oh))
            dec_res = eval_maps(dec_maps, gt_masks)
            dec_dices = np.array([r["dice"] for r in dec_res])

            if cond == "none":
                base_dec = dec_dices
            deltas_dec = dec_dices - base_dec
            ci_dec = bootstrap_ci(deltas_dec) if cond != "none" else (0.0, 0.0)

            dec_empty = [r for r in dec_res if r["is_empty"]]
            fp_dec = float(np.mean([r["fp_on_empty"] for r in dec_empty])) if dec_empty else 0.0

            all_dec.append({
                "site": site_name, "cond": cond, "n": n,
                "dice": float(dec_dices.mean()),
                "delta": float(deltas_dec.mean()),
                "CI_lo": ci_dec[0], "CI_hi": ci_dec[1],
                "n_empty": len(dec_empty),
                "fp_rate_empty": fp_dec,
            })

        del gt_masks, lr_maps, dec_maps

    # Print tables
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)

    df_lr = pd.DataFrame(all_lr)
    df_dec = pd.DataFrame(all_dec)
    cols = ["site", "cond", "n", "dice", "delta", "CI_lo", "CI_hi",
            "n_empty", "fp_rate_empty"]

    print("\n" + "="*120)
    print("TABLE A: LINEAR PROBE — CORAL + EM POC")
    print("="*120)
    print(df_lr[cols].to_string(index=False))

    print("\n" + "="*120)
    print("TABLE B: DECODER — CORAL + EM POC")
    print("="*120)
    print(df_dec[cols].to_string(index=False))

    # Summary: best condition per site
    print("\n" + "="*80)
    print("TABLE C: BEST CONDITION PER SITE")
    print("="*80)
    summary = []
    for site_name in [s[0] for s in sites]:
        lr_site = df_lr[df_lr["site"] == site_name]
        dec_site = df_dec[df_dec["site"] == site_name]
        lr_best = lr_site.loc[lr_site["dice"].idxmax()]
        dec_best = dec_site.loc[dec_site["dice"].idxmax()]
        summary.append({
            "site": site_name,
            "LR_best": lr_best["cond"], "LR_dice": lr_best["dice"],
            "Dec_best": dec_best["cond"], "Dec_dice": dec_best["dice"],
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # Save
    df_lr.to_csv(OUT_DIR / "lr_coral_poc.csv", index=False)
    df_dec.to_csv(OUT_DIR / "dec_coral_poc.csv", index=False)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
