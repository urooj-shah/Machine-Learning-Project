"""Audit experiment: corrected source prior + full metrics.

Tasks 1-7 from audit checklist. Key fix: LR pi_source = 0.5
(class_weight='balanced'), not y_train.mean().
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    GRID, N_PATCHES, bias_corrected_temperature_scale, bootstrap_ci,
    bbse_estimate, brier_score, dice_score, expected_calibration_error,
    load_gt_mask, load_site_features, logit_shift, nll_score,
    saerens_em, sigmoid, temperature_scale, upsample_probs,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "05_audit"
SEED = 42
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            "pred_fg": int(pb.sum()), "gt_fg": int(gb.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "is_empty": is_empty,
            "fp_on_empty": bool(is_empty and pb.sum() > 0),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
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

    # ── Train LR ───────────────────────────────────────────────────
    print("Training LR ...", flush=True)
    clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                             class_weight="balanced")
    X_train = kvasir["patch_tokens"][train_idx].reshape(-1, 384)
    y_train = kvasir["patch_fg_binary"][train_idx].reshape(-1)
    clf.fit(X_train, y_train)

    pi_source_lr = 0.5
    pi_natural = float(y_train.mean())

    # ═══════════════════════════════════════════════════════════════
    # TASK 1
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("TASK 1: SAERENS EM SOURCE PRIOR")
    print("="*70)
    print(f"  LR: class_weight='balanced' → pi_source = {pi_source_lr}")
    print(f"       natural training prior = {pi_natural:.4f} (NOT used)")
    print(f"       outputs/04_corrected/ used {pi_natural:.4f} → WRONG")

    # ── Source val ─────────────────────────────────────────────────
    X_val = kvasir["patch_tokens"][val_idx].reshape(-1, 384)
    y_val = kvasir["patch_fg_binary"][val_idx].reshape(-1)
    src_logits = clf.decision_function(X_val).astype(np.float64)
    src_probs = clf.predict_proba(X_val)[:, 1]

    # ═══════════════════════════════════════════════════════════════
    # TASK 2
    # ═══════════════════════════════════════════════════════════════
    temp_lr = temperature_scale(src_logits, y_val)
    ece_raw = expected_calibration_error(sigmoid(src_logits), y_val)
    ece_ts = expected_calibration_error(sigmoid(src_logits / temp_lr), y_val)

    T_bcts_lr, c_bcts_lr = bias_corrected_temperature_scale(src_logits, y_val)
    p_bcts_src = sigmoid(src_logits / T_bcts_lr + c_bcts_lr)
    pi_bcts_src_lr = float(p_bcts_src.mean())
    ece_bcts = expected_calibration_error(p_bcts_src, y_val)

    print("\n" + "="*70)
    print("TASK 2: TEMPERATURE SCALING")
    print("="*70)
    print("  Convention: scaled = logit / T")
    print("  T > 1 → overconfident   T < 1 → underconfident")
    print(f"  LR: T = {temp_lr:.4f} → {'over' if temp_lr > 1 else 'under'}confident")
    print(f"      ECE: {ece_raw:.4f} → {ece_ts:.4f} (TS) → {ece_bcts:.4f} (BCTS)")
    print(f"  LR BCTS: T={T_bcts_lr:.4f}, c={c_bcts_lr:.4f}, "
          f"pi_bcts_src={pi_bcts_src_lr:.4f}")

    # Reliability bins
    bins = np.linspace(0, 1, 11)
    probs_raw = sigmoid(src_logits)
    print(f"\n  {'bin':>12s}  {'pred':>6s}  {'true':>6s}  {'n':>8s}")
    for i in range(10):
        mask = (probs_raw > bins[i]) & (probs_raw <= bins[i+1])
        nn = int(mask.sum())
        if nn == 0:
            continue
        print(f"  ({bins[i]:.1f},{bins[i+1]:.1f}]  "
              f"{probs_raw[mask].mean():6.4f}  {y_val[mask].mean():6.4f}  {nn:8d}")

    # ── Train decoder ──────────────────────────────────────────────
    print("\nTraining decoder ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    pi_source_dec = float(kvasir_m128[train_idx].mean())
    t0 = time.time()
    decoder, dec_vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  Trained in {time.time()-t0:.0f}s, val Dice={dec_vd:.4f}")
    print(f"  Decoder pi_source = {pi_source_dec:.4f}")

    dec_logits_val = decoder_logits_128(decoder, kvasir["patch_tokens"][val_idx])
    dec_lf = dec_logits_val.reshape(-1).astype(np.float64)
    dec_lbl = kvasir_m128[val_idx].reshape(-1).astype(np.float64)
    temp_dec = temperature_scale(dec_lf, dec_lbl)
    T_bcts_dec, c_bcts_dec = bias_corrected_temperature_scale(dec_lf, dec_lbl)
    p_bcts_src_dec = sigmoid(dec_lf / T_bcts_dec + c_bcts_dec)
    pi_bcts_src_dec = float(p_bcts_src_dec.mean())

    print(f"  Dec: T={temp_dec:.4f} → {'over' if temp_dec > 1 else 'under'}confident")
    print(f"  Dec BCTS: T={T_bcts_dec:.4f}, c={c_bcts_dec:.4f}, "
          f"pi_bcts_src={pi_bcts_src_dec:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # TASKS 3-6: EVALUATE ALL SITES
    # ═══════════════════════════════════════════════════════════════
    sites = [("kvasir_val", kvasir, val_idx)]
    for nm in TARGET_SITES:
        sites.append((nm, targets[nm], np.arange(len(targets[nm]["filenames"]))))

    methods = ["no_correction", "oracle", "em", "bcts_em", "bbse"]

    all_lr, all_dec = [], []
    prior_rows = []
    thresh_rows = []

    for site_name, sd, indices in sites:
        n = len(indices)
        print(f"\n  Evaluating {site_name} ({n}) ...", flush=True)

        gt_masks = [load_gt_mask(str(sd["mask_paths"][idx])) for idx in indices]
        m128 = load_masks_128(sd["mask_paths"][indices])

        # LR predictions
        Xs = sd["patch_tokens"][indices].reshape(-1, 384)
        lr_p = clf.predict_proba(Xs)[:, 1].astype(np.float32)
        lr_l = clf.decision_function(Xs).astype(np.float64)
        lr_p_img = [lr_p[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]
        lr_l_img = [lr_l[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]

        # Decoder predictions
        dl_all = decoder_logits_128(decoder, sd["patch_tokens"][indices])
        dp_all = sigmoid(dl_all)

        # Priors
        pi_true_p = float(sd["patch_fg_binary"][indices].reshape(-1).mean())
        pi_true_px = float(m128.mean())
        lr_flat = lr_p.astype(np.float64)
        dec_flat = dp_all.reshape(-1).astype(np.float64)

        pi_em_lr = saerens_em(lr_flat, pi_source=pi_source_lr)
        pi_em_dec = saerens_em(dec_flat, pi_source=pi_source_dec)

        lr_bcts_f = sigmoid(lr_l / T_bcts_lr + c_bcts_lr)
        pi_bcts_em_lr = saerens_em(lr_bcts_f, pi_source=pi_bcts_src_lr)

        dl_site_f = dl_all.reshape(-1).astype(np.float64)
        dec_bcts_f = sigmoid(dl_site_f / T_bcts_dec + c_bcts_dec)
        pi_bcts_em_dec = saerens_em(dec_bcts_f, pi_source=pi_bcts_src_dec)

        pi_bbse_lr = bbse_estimate(
            (src_probs >= 0.5).astype(int), y_val.astype(int),
            (lr_flat >= 0.5).astype(int))
        pi_bbse_dec = bbse_estimate(
            (sigmoid(dec_lf) >= 0.5).astype(int), dec_lbl.astype(int),
            (dec_flat >= 0.5).astype(int))

        prior_rows.append({
            "site": site_name,
            "pi_true_patch": pi_true_p, "pi_true_pixel": pi_true_px,
            "lr_em": pi_em_lr, "lr_bcts_em": pi_bcts_em_lr,
            "lr_bbse": pi_bbse_lr,
            "dec_em": pi_em_dec, "dec_bcts_em": pi_bcts_em_dec,
            "dec_bbse": pi_bbse_dec,
        })

        # Shifts
        s_lr = {
            "oracle": logit_shift(pi_source_lr, pi_true_p),
            "em": logit_shift(pi_source_lr, pi_em_lr),
            "bbse": logit_shift(pi_source_lr, pi_bbse_lr),
        }
        s_dec = {
            "oracle": logit_shift(pi_source_dec, pi_true_px),
            "em": logit_shift(pi_source_dec, pi_em_dec),
            "bbse": logit_shift(pi_source_dec, pi_bbse_dec),
        }
        s_bcts_lr = logit_shift(pi_bcts_src_lr, pi_bcts_em_lr)
        s_bcts_dec = logit_shift(pi_bcts_src_dec, pi_bcts_em_dec)

        # Build corrected maps
        lr_maps, dec_maps = {}, {}
        lr_cflat, dec_cflat = {}, {}

        # no_correction
        lm, dm = [], []
        for i, idx in enumerate(indices):
            ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
            lm.append(up_lr(lr_p_img[i], ow, oh))
            dm.append(up_dec(dp_all[i], ow, oh))
        lr_maps["no_correction"] = lm
        dec_maps["no_correction"] = dm
        lr_cflat["no_correction"] = lr_flat
        dec_cflat["no_correction"] = dec_flat

        for mt in ["oracle", "em", "bbse"]:
            lm, dm, lcp, dcp = [], [], [], []
            for i, idx in enumerate(indices):
                ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
                lc = sigmoid(lr_l_img[i] + s_lr[mt]).astype(np.float32)
                lm.append(up_lr(lc, ow, oh))
                lcp.append(lc)
                dc = sigmoid(dl_all[i] + s_dec[mt]).astype(np.float32)
                dm.append(up_dec(dc, ow, oh))
                dcp.append(dc.flatten())
            lr_maps[mt] = lm; dec_maps[mt] = dm
            lr_cflat[mt] = np.concatenate(lcp).astype(np.float64)
            dec_cflat[mt] = np.concatenate(dcp).astype(np.float64)

        # bcts_em
        lm, dm, lcp, dcp = [], [], [], []
        for i, idx in enumerate(indices):
            ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
            bl = lr_l_img[i] / T_bcts_lr + c_bcts_lr + s_bcts_lr
            lc = sigmoid(bl).astype(np.float32)
            lm.append(up_lr(lc, ow, oh))
            lcp.append(lc)
            bd = dl_all[i].flatten().astype(np.float64) / T_bcts_dec + c_bcts_dec + s_bcts_dec
            dc = sigmoid(bd).astype(np.float32)
            dm.append(up_dec(dc.reshape(MASK_SIZE, MASK_SIZE), ow, oh))
            dcp.append(dc)
        lr_maps["bcts_em"] = lm; dec_maps["bcts_em"] = dm
        lr_cflat["bcts_em"] = np.concatenate(lcp).astype(np.float64)
        dec_cflat["bcts_em"] = np.concatenate(dcp).astype(np.float64)

        # Evaluate
        lr_labels = sd["patch_fg_binary"][indices].reshape(-1).astype(np.float64)
        dec_labels = m128.reshape(-1).astype(np.float64)

        none_d_lr = none_d_dec = None

        for method in methods:
            rl = eval_maps(lr_maps[method], gt_masks)
            rd = eval_maps(dec_maps[method], gt_masks)

            for tag, res, cflat, labels, out, nm_d_ref in [
                ("linear_probe", rl, lr_cflat[method], lr_labels, all_lr, "lr"),
                ("decoder", rd, dec_cflat[method], dec_labels, all_dec, "dec"),
            ]:
                dices = np.array([r["dice"] for r in res])
                if method == "no_correction":
                    if nm_d_ref == "lr":
                        none_d_lr = dices
                    else:
                        none_d_dec = dices

                ref = none_d_lr if nm_d_ref == "lr" else none_d_dec
                deltas = dices - ref if ref is not None else np.zeros_like(dices)
                ci = bootstrap_ci(deltas) if method != "no_correction" and ref is not None else (0.0, 0.0)

                tp_s = sum(r["tp"] for r in res)
                fp_s = sum(r["fp"] for r in res)
                fn_s = sum(r["fn"] for r in res)
                pooled = 2*tp_s / max(2*tp_s+fp_s+fn_s, 1)

                d_pos = [r["dice"] for r in res if not r["is_empty"]]
                d_emp = [r["dice"] for r in res if r["is_empty"]]
                fp_e = [r["fp_on_empty"] for r in res if r["is_empty"]]

                cf = np.clip(cflat, 1e-10, 1-1e-10)

                out.append({
                    "site": site_name, "method": method, "n": n,
                    "dice": float(dices.mean()),
                    "pooled": pooled,
                    "prec": float(np.mean([r["prec"] for r in res])),
                    "rec": float(np.mean([r["rec"] for r in res])),
                    "pred_fg": float(np.mean([r["pred_fg"] for r in res])),
                    "gt_fg": float(np.mean([r["gt_fg"] for r in res])),
                    "ECE": expected_calibration_error(cf, labels),
                    "NLL": nll_score(cf, labels),
                    "Brier": brier_score(cf, labels),
                    "delta": float(deltas.mean()),
                    "CI_lo": ci[0], "CI_hi": ci[1],
                    "dice_pos": float(np.mean(d_pos)) if d_pos else float("nan"),
                    "n_pos": len(d_pos),
                    "dice_empty": float(np.mean(d_emp)) if d_emp else float("nan"),
                    "n_empty": len(d_emp),
                    "fp_rate_empty": float(np.mean(fp_e)) if fp_e else 0.0,
                })

        # Threshold sweep at 128×128 resolution for speed
        gt128 = [(m128[i] >= 0.5).astype(np.uint8) for i in range(n)]
        # LR probs at 128×128
        lr128 = {}
        lr128["no_correction"] = [upsample_probs(
            lr_p_img[i].reshape(GRID, GRID), MASK_SIZE, MASK_SIZE)
            for i in range(n)]
        for mt in ["oracle", "em"]:
            lr128[mt] = [upsample_probs(
                sigmoid(lr_l_img[i] + s_lr[mt]).astype(np.float32).reshape(GRID, GRID),
                MASK_SIZE, MASK_SIZE) for i in range(n)]
        # Decoder probs at 128×128 (native resolution)
        dec128 = {}
        dec128["no_correction"] = [dp_all[i] for i in range(n)]
        for mt in ["oracle", "em"]:
            dec128[mt] = [sigmoid(dl_all[i] + s_dec[mt]).astype(np.float32)
                          for i in range(n)]

        for t in np.arange(0.1, 0.95, 0.1):
            tv = round(float(t), 2)
            for mt in ["no_correction", "oracle", "em"]:
                d_lr = [dice_score((m >= tv).astype(np.uint8), g)
                        for m, g in zip(lr128[mt], gt128)]
                d_dec = [dice_score((m >= tv).astype(np.uint8), g)
                         for m, g in zip(dec128[mt], gt128)]
                thresh_rows.append({"site": site_name, "model": "LR",
                                    "method": mt, "t": tv,
                                    "dice": float(np.mean(d_lr))})
                thresh_rows.append({"site": site_name, "model": "Dec",
                                    "method": mt, "t": tv,
                                    "dice": float(np.mean(d_dec))})
        del lr128, dec128, gt128

        del gt_masks, lr_maps, dec_maps

    # ═══════════════════════════════════════════════════════════════
    # PRINT TABLES
    # ═══════════════════════════════════════════════════════════════
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    df_lr = pd.DataFrame(all_lr)
    df_dec = pd.DataFrame(all_dec)

    c3 = ["site", "method", "n", "dice", "pooled", "prec", "rec",
          "pred_fg", "gt_fg", "ECE", "NLL", "Brier",
          "delta", "CI_lo", "CI_hi"]

    print("\n" + "="*130)
    print("TABLE 3a: LINEAR PROBE (pi_source=0.5)")
    print("="*130)
    print(df_lr[c3].to_string(index=False))

    print("\n" + "="*130)
    print("TABLE 3b: DECODER (pi_source={:.4f})".format(pi_source_dec))
    print("="*130)
    print(df_dec[c3].to_string(index=False))

    # TABLE 4a
    c4 = ["site", "method", "n_pos", "dice_pos", "n_empty",
          "dice_empty", "fp_rate_empty"]
    print("\n" + "="*100)
    print("TABLE 4a: POSITIVES-ONLY DICE / EMPTY-IMAGE FP")
    print("="*100)
    print("\n  LINEAR PROBE:")
    print(df_lr[c4].to_string(index=False))
    print("\n  DECODER:")
    print(df_dec[c4].to_string(index=False))

    # TABLE 4b: C4 decomposition
    print("\n" + "="*100)
    print("TABLE 4b: C4 ORACLE GAIN DECOMPOSITION")
    print("="*100)
    for tag, df in [("LR", df_lr), ("Decoder", df_dec)]:
        c4d = df[df["site"] == "polypgen_C4"]
        nc = c4d[c4d["method"] == "no_correction"].iloc[0]
        oc = c4d[c4d["method"] == "oracle"].iloc[0]
        n_tot = nc["n"]
        n_p, n_e = nc["n_pos"], nc["n_empty"]
        g_all = oc["dice"] - nc["dice"]
        g_pos = (oc["dice_pos"] - nc["dice_pos"]) * n_p / n_tot if n_p > 0 else 0
        g_emp = (oc["dice_empty"] - nc["dice_empty"]) * n_e / n_tot if n_e > 0 else 0
        print(f"\n  {tag}: n={n_tot}, n_pos={n_p}, n_empty={n_e}")
        print(f"    none:   dice_pos={nc['dice_pos']:.4f}  "
              f"dice_empty={nc['dice_empty']:.4f}  dice_all={nc['dice']:.4f}")
        print(f"    oracle: dice_pos={oc['dice_pos']:.4f}  "
              f"dice_empty={oc['dice_empty']:.4f}  dice_all={oc['dice']:.4f}")
        print(f"    gain_total={g_all:.4f}  gain_pos={g_pos:.4f}  gain_empty={g_emp:.4f}")

    # TABLE 5
    print("\n" + "="*130)
    print("TABLE 5: DICE vs THRESHOLD — best per site (upper bound)")
    print("="*130)
    df_t = pd.DataFrame(thresh_rows)
    best = df_t.loc[df_t.groupby(["site", "model", "method"])["dice"].idxmax()]
    print(best[["site", "model", "method", "t", "dice"]].to_string(index=False))

    print("\n  LR full sweep:")
    lr_t = df_t[df_t["model"] == "LR"]
    for mt in ["no_correction", "oracle", "em"]:
        sub = lr_t[lr_t["method"] == mt]
        piv = sub.pivot_table(index="site", columns="t", values="dice")
        print(f"\n    {mt}:")
        print(piv.round(4).to_string())

    # TABLE 6
    print("\n" + "="*100)
    print("TABLE 6: PRIOR ESTIMATES — estimated vs true")
    print(f"  LR BCTS: T={T_bcts_lr:.4f}, c={c_bcts_lr:.4f}")
    print(f"  Dec BCTS: T={T_bcts_dec:.4f}, c={c_bcts_dec:.4f}")
    print("="*100)
    df_p = pd.DataFrame(prior_rows)
    df_p["lr_em_err"] = (df_p["lr_em"] - df_p["pi_true_patch"]).abs()
    df_p["lr_bcts_err"] = (df_p["lr_bcts_em"] - df_p["pi_true_patch"]).abs()
    df_p["lr_bbse_err"] = (df_p["lr_bbse"] - df_p["pi_true_patch"]).abs()
    df_p["dec_em_err"] = (df_p["dec_em"] - df_p["pi_true_pixel"]).abs()
    df_p["dec_bcts_err"] = (df_p["dec_bcts_em"] - df_p["pi_true_pixel"]).abs()
    df_p["dec_bbse_err"] = (df_p["dec_bbse"] - df_p["pi_true_pixel"]).abs()

    print("\n  LR:")
    print(df_p[["site", "pi_true_patch", "lr_em", "lr_em_err",
                "lr_bcts_em", "lr_bcts_err", "lr_bbse", "lr_bbse_err"]].to_string(index=False))
    print("\n  Decoder:")
    print(df_p[["site", "pi_true_pixel", "dec_em", "dec_em_err",
                "dec_bcts_em", "dec_bcts_err", "dec_bbse", "dec_bbse_err"]].to_string(index=False))

    # Save
    df_lr.to_csv(OUT_DIR / "lr_results.csv", index=False)
    df_dec.to_csv(OUT_DIR / "dec_results.csv", index=False)
    df_p.to_csv(OUT_DIR / "prior_estimates.csv", index=False)
    df_t.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    print(f"\nTasks 1-6 saved to {OUT_DIR}/")
    print("Task 7 (few-label ablation): run scripts/fewlabel_ablation.py")
    print("Task 8 (regression tests): python -m pytest tests/ -v")
    print("DONE")


if __name__ == "__main__":
    main()
