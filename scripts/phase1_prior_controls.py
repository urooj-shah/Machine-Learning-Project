"""Phase 1: Controls on prior-correction result.

Evaluates at 128x128 resolution (native decoder output) for speed.
Three heads: balanced LR, unbalanced LR, decoder.
Five correction variants per head.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import (
    GRID, N_PATCHES, bootstrap_ci, dice_score,
    load_site_features, logit_shift, saerens_em, sigmoid,
    temperature_scale, upsample_probs,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CKPT_PATH = ROOT / "checkpoints" / "decoder_corrected.pt"
OUT_DIR = ROOT / "outputs" / "06_poc"
SEED = 42
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


def load_masks_128(mask_paths):
    masks = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
        masks.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(masks)


@torch.no_grad()
def decoder_logits_128(model, tokens):
    model.eval()
    out = []
    for i in range(0, len(tokens), 32):
        batch = tokens[i:i + 32]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_at_128(probs_flat, gt_128, n, res_size, is_lr=False, threshold=0.5):
    """Evaluate at 128x128 resolution. For LR: upsample 16x16->128x128."""
    rows = []
    for i in range(n):
        if is_lr:
            p16 = probs_flat[i * N_PATCHES:(i + 1) * N_PATCHES].reshape(GRID, GRID)
            p128 = np.array(Image.fromarray(p16.astype(np.float32), mode="F").resize(
                (MASK_SIZE, MASK_SIZE), Image.BILINEAR))
        else:
            p128 = probs_flat[i * MASK_SIZE * MASK_SIZE:(i + 1) * MASK_SIZE * MASK_SIZE].reshape(MASK_SIZE, MASK_SIZE)

        pred = (p128 >= threshold).astype(np.uint8)
        gt = gt_128[i]
        pb, gb = pred.astype(bool), (gt >= 0.5).astype(bool)
        tp = int((pb & gb).sum())
        fp = int((pb & ~gb).sum())
        fn = int((~pb & gb).sum())
        is_empty = (gb.sum() == 0)
        rows.append({
            "dice": dice_score(pred, gt),
            "tp": tp, "fp": fp, "fn": fn,
            "is_empty": is_empty,
            "fp_on_empty": bool(is_empty and pb.sum() > 0),
        })
    return rows


def compute_metrics(res_list, baseline_dices=None):
    dices = np.array([r["dice"] for r in res_list])
    pos_dices = np.array([r["dice"] for r in res_list if not r["is_empty"]])
    empty = [r for r in res_list if r["is_empty"]]
    fp_e = [r["fp_on_empty"] for r in empty] if empty else []
    tp_s = sum(r["tp"] for r in res_list)
    fp_s = sum(r["fp"] for r in res_list)
    fn_s = sum(r["fn"] for r in res_list)
    pooled = 2 * tp_s / max(2 * tp_s + fp_s + fn_s, 1)

    if baseline_dices is not None:
        deltas = dices - baseline_dices
        ci = bootstrap_ci(deltas)
    else:
        deltas = np.zeros_like(dices)
        ci = (0.0, 0.0)

    return {
        "dice": float(dices.mean()),
        "pooled": pooled,
        "dice_pos": float(pos_dices.mean()) if len(pos_dices) > 0 else float("nan"),
        "n_pos": len(pos_dices),
        "n_empty": len(empty),
        "fp_rate_empty": float(np.mean(fp_e)) if fp_e else 0.0,
        "delta": float(deltas.mean()),
        "CI_lo": ci[0], "CI_hi": ci[1],
    }


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

    X_train = kvasir["patch_tokens"][train_idx].reshape(-1, 384)
    y_train = kvasir["patch_fg_binary"][train_idx].reshape(-1)
    X_val = kvasir["patch_tokens"][val_idx].reshape(-1, 384)
    y_val = kvasir["patch_fg_binary"][val_idx].reshape(-1)

    print("Training LR models ...", flush=True)
    clf_bal = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                                 class_weight="balanced")
    clf_bal.fit(X_train, y_train)

    clf_unb = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                                 class_weight=None)
    clf_unb.fit(X_train, y_train)
    pi_unb = float(y_train.mean())
    print(f"  Natural prior: {pi_unb:.4f}")

    print("Loading decoder ...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    decoder = PatchDecoder(in_dim=384).to(DEVICE)
    decoder.load_state_dict(ckpt["model"])
    pi_dec = float(ckpt["pi_source"])

    # Cache all masks at 128x128
    print("Caching masks at 128x128 ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    target_m128 = {}
    for name in TARGET_SITES:
        target_m128[name] = load_masks_128(targets[name]["mask_paths"])

    # Temperature scaling on val
    bal_val_l = clf_bal.decision_function(X_val).astype(np.float64)
    T_bal = temperature_scale(bal_val_l, y_val)
    unb_val_l = clf_unb.decision_function(X_val).astype(np.float64)
    T_unb = temperature_scale(unb_val_l, y_val)

    dec_val_l = decoder_logits_128(decoder, kvasir["patch_tokens"][val_idx])
    dec_val_flat = dec_val_l.reshape(-1).astype(np.float64)
    dec_val_lbl = kvasir_m128[val_idx].reshape(-1).astype(np.float64)
    T_dec = temperature_scale(dec_val_flat, dec_val_lbl)
    print(f"  T_bal={T_bal:.4f}, T_unb={T_unb:.4f}, T_dec={T_dec:.4f}")

    # Threshold grid
    logit_grid = np.linspace(-4.6, 4.6, 31)
    thresh_grid = sigmoid(logit_grid)

    sites = [("kvasir_val", kvasir, val_idx, kvasir_m128[val_idx])]
    for nm in TARGET_SITES:
        sites.append((nm, targets[nm], np.arange(len(targets[nm]["filenames"])), target_m128[nm]))

    heads = [
        ("LR_balanced", clf_bal, 0.5, T_bal, True),
        ("LR_unbalanced", clf_unb, pi_unb, T_unb, True),
        ("Decoder", None, pi_dec, T_dec, False),
    ]

    methods = ["baseline", "oracle", "em", "temp_only", "source_prior"]
    all_rows = []
    thresh_rows = []

    for site_name, sd, indices, gt_128 in sites:
        n = len(indices)
        print(f"\n  {site_name} ({n}) ...", flush=True)

        pi_true_patch = float(sd["patch_fg_binary"][indices].reshape(-1).mean())
        pi_true_pixel = float(gt_128.mean())

        for head_name, clf, pi_src, T, is_lr in heads:
            if is_lr:
                Xs = sd["patch_tokens"][indices].reshape(-1, 384)
                logits_flat = clf.decision_function(Xs).astype(np.float64)
                pi_true = pi_true_patch
            else:
                dl_128 = decoder_logits_128(decoder, sd["patch_tokens"][indices])
                logits_flat = dl_128.reshape(-1).astype(np.float64)
                pi_true = pi_true_pixel

            probs_flat = sigmoid(logits_flat).astype(np.float64)
            pi_em = saerens_em(probs_flat, pi_source=pi_src)

            # Threshold sweep on base logits
            for t_val in thresh_grid:
                base_probs = sigmoid(logits_flat).astype(np.float32)
                res_t = eval_at_128(base_probs, gt_128, n, MASK_SIZE, is_lr=is_lr,
                                     threshold=float(t_val))
                dices_t = np.array([r["dice"] for r in res_t])
                pos_dices_t = np.array([r["dice"] for r in res_t if not r["is_empty"]])
                thresh_rows.append({
                    "head": head_name, "site": site_name,
                    "threshold": round(float(t_val), 4),
                    "dice": float(dices_t.mean()),
                    "dice_pos": float(pos_dices_t.mean()) if len(pos_dices_t) > 0 else float("nan"),
                })

            # Correction variants
            shifts = {
                "baseline": 0.0,
                "oracle": logit_shift(pi_src, pi_true),
                "em": logit_shift(pi_src, pi_em),
                "temp_only": 0.0,
                "source_prior": 0.0,
            }

            baseline_dices = None
            for method in methods:
                s = shifts[method]
                if method == "temp_only":
                    corrected_probs = sigmoid(logits_flat / T).astype(np.float32)
                else:
                    corrected_probs = sigmoid(logits_flat + s).astype(np.float32)

                res = eval_at_128(corrected_probs, gt_128, n, MASK_SIZE, is_lr=is_lr)
                dices = np.array([r["dice"] for r in res])

                if method == "baseline":
                    baseline_dices = dices.copy()

                metrics = compute_metrics(res, baseline_dices if method != "baseline" else None)
                implied_thresh = sigmoid(-s) if method in ("oracle", "em") else float("nan")

                all_rows.append({
                    "head": head_name, "site": site_name, "method": method, "n": n,
                    "dice": metrics["dice"], "pooled": metrics["pooled"],
                    "dice_pos": metrics["dice_pos"],
                    "n_pos": metrics["n_pos"], "n_empty": metrics["n_empty"],
                    "fp_rate_empty": metrics["fp_rate_empty"],
                    "delta": metrics["delta"],
                    "CI_lo": metrics["CI_lo"], "CI_hi": metrics["CI_hi"],
                    "pi_true": pi_true, "pi_est": pi_em if method == "em" else float("nan"),
                    "shift": s, "implied_thresh": implied_thresh,
                })

    # Print tables
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 25)

    df = pd.DataFrame(all_rows)
    df_t = pd.DataFrame(thresh_rows)

    cols = ["head", "site", "method", "dice", "pooled", "dice_pos",
            "n_pos", "n_empty", "fp_rate_empty",
            "delta", "CI_lo", "CI_hi", "implied_thresh"]

    print("\n" + "=" * 180)
    print("TABLE 1: PRIOR-CORRECTION HEADROOM (128x128 eval)")
    print("=" * 180)
    print(df[cols].to_string(index=False))

    # Best threshold
    print("\n" + "=" * 120)
    print("TABLE 2: BEST THRESHOLD PER SITE (headroom over t=0.5)")
    print("=" * 120)
    best_t = df_t.loc[df_t.groupby(["head", "site"])["dice"].idxmax()].copy()
    t05_mask = df_t["threshold"].between(0.48, 0.52)
    t05 = df_t[t05_mask].copy()
    if len(t05) > 0:
        t05_best = t05.loc[t05.groupby(["head", "site"])["dice"].idxmax()]
        merged = best_t.merge(t05_best[["head", "site", "dice"]],
                              on=["head", "site"], suffixes=("_best", "_t05"))
        merged["headroom"] = merged["dice_best"] - merged["dice_t05"]
        print(merged[["head", "site", "threshold", "dice_best", "dice_t05", "headroom"]].to_string(index=False))

    # Prior estimation
    print("\n" + "=" * 100)
    print("TABLE 3: PRIOR ESTIMATES (EM)")
    print("=" * 100)
    em_df = df[df["method"] == "em"][["head", "site", "pi_true", "pi_est"]].copy()
    em_df["error"] = (em_df["pi_est"] - em_df["pi_true"]).abs()
    print(em_df.to_string(index=False))

    df.to_csv(OUT_DIR / "phase1_headroom.csv", index=False)
    df_t.to_csv(OUT_DIR / "phase1_threshold_sweep.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
