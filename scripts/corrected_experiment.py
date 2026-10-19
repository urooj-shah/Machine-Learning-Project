"""Corrected prior-correction experiment: linear probe + decoder.

Supersedes prior_correction_experiment.py and decoder_prior_experiment.py.
All metrics from src/eval.py. Caches masks and predictions to avoid
redundant I/O and recomputation.
"""

import json
import os
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import (
    GRID, N_PATCHES, bootstrap_ci, bbse_estimate, brier_score,
    dice_score, expected_calibration_error, iou_score, load_gt_mask,
    load_site_features, logit_shift, nll_score, pixel_metrics,
    pixel_metrics_128, saerens_em, sigmoid, temperature_scale,
    upsample_probs,
)
from patch_decoder import PatchDecoder

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs" / "04_corrected"
SEED = 42
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


# ── Dataset for decoder ────────────────────────────────────────────

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


# ── Decoder training ───────────────────────────────────────────────

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder_fn(train_tokens, train_masks, val_tokens=None, val_masks=None,
                     n_epochs=80, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatureDataset(train_tokens, train_masks)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    model = PatchDecoder(in_dim=384).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce_fn = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = bce_fn(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if val_tokens is not None:
            model.eval()
            val_ds = FeatureDataset(val_tokens, val_masks)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
            total_dice = 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    preds = (torch.sigmoid(model(x)) >= 0.5).float()
                    smooth = 1.0
                    inter = (preds * y).sum(dim=(1, 2, 3))
                    union = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                    total_dice += ((2 * inter + smooth) / (union + smooth)).sum().item()
            val_dice = total_dice / len(val_ds)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pi_source = float(train_masks.mean())
    return model, pi_source, best_val_dice


@torch.no_grad()
def get_decoder_logits_128(model, patch_tokens):
    model.eval()
    all_logits = []
    for i in range(0, len(patch_tokens), 32):
        batch = patch_tokens[i:i+32]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out = model(x).cpu().numpy()[:, 0]
        all_logits.append(out)
    return np.concatenate(all_logits)


# ── Cached evaluation (load GT masks ONCE per site) ────────────────

def load_gt_masks_cached(mask_paths, orig_sizes):
    """Load all full-resolution GT masks for a site. Returns list of arrays."""
    masks = []
    for i in range(len(mask_paths)):
        masks.append(load_gt_mask(str(mask_paths[i])))
    return masks


def upsample_pred_fullres(prob_256, orig_w, orig_h):
    """Upsample 256 patch probs to full resolution."""
    prob_grid = np.asarray(prob_256, dtype=np.float32).reshape(GRID, GRID)
    return upsample_probs(prob_grid, orig_w, orig_h)


def upsample_pred_128(prob_128, orig_w, orig_h):
    """Upsample 128x128 prob map to full resolution."""
    arr = np.asarray(prob_128, dtype=np.float32)
    img = Image.fromarray(arr, mode="F")
    return np.array(img.resize((int(orig_w), int(orig_h)), Image.BILINEAR))


def evaluate_cached(prob_maps_up, gt_masks, threshold=0.5):
    """Evaluate Dice/IoU on pre-upsampled probability maps and cached GT masks."""
    results = []
    for prob_up, gt in zip(prob_maps_up, gt_masks):
        pred = (prob_up >= threshold).astype(np.uint8)
        pred_b = pred.astype(bool)
        gt_b = gt.astype(bool)
        tp = int((pred_b & gt_b).sum())
        fp = int((pred_b & ~gt_b).sum())
        fn = int((~pred_b & gt_b).sum())
        results.append({
            "dice": dice_score(pred, gt),
            "iou": iou_score(pred, gt),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "pred_fg": int(pred_b.sum()),
            "gt_fg": int(gt_b.sum()),
            "fp_on_empty": bool(gt_b.sum() == 0 and pred_b.sum() > 0),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────
    print("Loading Kvasir ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}", flush=True)

    targets = {}
    for name in TARGET_SITES:
        targets[name] = load_site_features(FEAT_DIR, name)
        print(f"  {name}: {len(targets[name]['filenames'])} images", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: VERIFICATION TESTS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: VERIFICATION TESTS", flush=True)
    print("=" * 70, flush=True)

    # 2a: GT patches → quantization ceiling
    print("\n--- 2a: GT patches as predictions → quantization ceiling ---", flush=True)
    dices_ceil = []
    for idx in val_idx:
        gt_patches = kvasir["patch_fg_binary"][idx].astype(np.float32)
        m = pixel_metrics(gt_patches, str(kvasir["mask_paths"][idx]),
                          int(kvasir["orig_sizes"][idx][0]),
                          int(kvasir["orig_sizes"][idx][1]))
        dices_ceil.append(m["dice"])
    ceil = np.mean(dices_ceil)
    print(f"  Kvasir val ceiling: {ceil:.4f} (expected ~0.916)", flush=True)
    print(f"  {'PASS' if abs(ceil - 0.916) < 0.01 else 'FAIL'}", flush=True)

    # 2b: Full-res GT → Dice 1.0
    print("\n--- 2b: Full-res GT mask as prediction → Dice = 1.0 ---", flush=True)
    ok2b = True
    for idx in val_idx[:10]:
        gt = load_gt_mask(str(kvasir["mask_paths"][idx]))
        if dice_score(gt, gt) != 1.0:
            ok2b = False
    print(f"  {'PASS' if ok2b else 'FAIL'}", flush=True)

    # 2c: Linear probe — patch-grid vs full-res Dice
    print("\n--- 2c: Linear probe Dice types ---", flush=True)
    clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                             class_weight="balanced")
    X_train = kvasir["patch_tokens"][train_idx].reshape(-1, 384)
    y_train = kvasir["patch_fg_binary"][train_idx].reshape(-1)
    clf.fit(X_train, y_train)
    pi_source_lr = float(y_train.mean())

    patch_dices, fullres_dices = [], []
    for idx in val_idx:
        X_img = kvasir["patch_tokens"][idx].reshape(N_PATCHES, -1)
        probs = clf.predict_proba(X_img)[:, 1]
        patch_dices.append(dice_score((probs >= 0.5).astype(int), kvasir["patch_fg_binary"][idx]))
        m = pixel_metrics(probs, str(kvasir["mask_paths"][idx]),
                          int(kvasir["orig_sizes"][idx][0]),
                          int(kvasir["orig_sizes"][idx][1]))
        fullres_dices.append(m["dice"])

    print(f"  Patch-grid Dice (16x16): {np.mean(patch_dices):.4f}", flush=True)
    print(f"  Full-res Dice:           {np.mean(fullres_dices):.4f}", flush=True)
    ok2c = abs(np.mean(fullres_dices) - 0.765) < 0.02
    print(f"  Full-res ~0.765: {'PASS' if ok2c else 'FAIL'} ({np.mean(fullres_dices):.4f})", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: BUG DIFF REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: BUG DIFF", flush=True)
    print("=" * 70, flush=True)
    print("""
BUG 1 (ROOT CAUSE of 0.218 Dice):
  prior_correction_experiment.py:66  Image.fromarray(prob_grid, mode="F")
  sklearn.predict_proba() returns float64. PIL mode 'F' expects float32.
  Float64 bytes misread as float32 → garbage values (±10^38) → random mask.
  diagnose_dice.py had .astype(np.float32) → correct.

BUG 2 (decoder Dice 0.549 < linear probe 0.765):
  train_decoder.py:41  tokens.reshape(384, 16, 16)  ← WRONG
  Correct: tokens.reshape(16, 16, 384).transpose(2, 0, 1)
  Mixes features and spatial positions — conv filters can't learn properly.

BUG 3 (decoder evaluation):
  decoder_prior_experiment.py:82  Image.fromarray(prob_128, mode="F")
  sigmoid() returns float64. Same PIL misinterpretation as Bug 1.

BUG 4 (Saerens EM source prior):
  saerens_em uses pi_s = probs.mean() instead of training-set prior.
  Biases EM estimate.
""", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: PAIRING VERIFICATION
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70, flush=True)
    print("STEP 4: IMAGE/MASK PAIRING", flush=True)
    print("=" * 70, flush=True)
    import yaml
    with open(ROOT / "configs" / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)
    data_root = ROOT / cfg["data_root"]
    for ds_name, ds_cfg in cfg["datasets"].items():
        img_dir = data_root / ds_cfg["images"]
        mask_dir = data_root / ds_cfg["masks"]
        pattern = ds_cfg["mask_pattern"]
        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))])
        matched, unmatched = 0, []
        for fname in imgs:
            stem = Path(fname).stem
            if (mask_dir / pattern.format(stem=stem)).exists():
                matched += 1
            else:
                unmatched.append(fname)
        # Check first 20 aspect ratios
        ar_ok = True
        for fname in imgs[:20]:
            stem = Path(fname).stem
            mask_name = pattern.format(stem=stem)
            mp = mask_dir / mask_name
            if mp.exists():
                if Image.open(img_dir / fname).size != Image.open(mp).size:
                    ar_ok = False
        # npz count
        npz = FEAT_DIR / f"{ds_name}.npz"
        n_npz = len(np.load(npz)["filenames"]) if npz.exists() else "?"
        print(f"  {ds_name:15s}: {matched:4d} pairs, {len(unmatched)} unmatched, "
              f"AR {'OK' if ar_ok else 'MISMATCH'}, npz={n_npz}", flush=True)
        for u in unmatched:
            print(f"    unmatched: {u}", flush=True)

    print("\n  C3 filename typo: 'C3_EndoCV2021_00489].jpg' has extra ']'.", flush=True)
    print("  Mask not found → skipped during extraction. No label misalignment.", flush=True)
    print("  C3 npz has 456 files = 457 images − 1 skip.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: TRAIN & EVALUATE (corrected)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("STEP 5: CORRECTED EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # Source val data for calibration + BBSE
    X_val = kvasir["patch_tokens"][val_idx].reshape(-1, 384)
    y_val = kvasir["patch_fg_binary"][val_idx].reshape(-1)
    source_logits = clf.decision_function(X_val)
    source_probs = clf.predict_proba(X_val)[:, 1]

    # Temperature scaling (Step 7)
    print("\nTemperature scaling on Kvasir val ...", flush=True)
    temp_lr = temperature_scale(source_logits, y_val)
    ece_before = expected_calibration_error(sigmoid(source_logits), y_val)
    ece_after = expected_calibration_error(sigmoid(source_logits / temp_lr), y_val)
    print(f"  T={temp_lr:.4f}, ECE: {ece_before:.4f} → {ece_after:.4f}", flush=True)

    # Train decoder (corrected reshape)
    print("\nTraining decoder (corrected reshape) ...", flush=True)
    kvasir_masks_128 = load_masks_128(kvasir["mask_paths"])
    t0 = time.time()
    decoder, pi_source_dec, dec_val_dice = train_decoder_fn(
        kvasir["patch_tokens"][train_idx], kvasir_masks_128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_masks_128[val_idx],
        n_epochs=80, seed=SEED,
    )
    print(f"  Trained in {time.time()-t0:.0f}s, val Dice={dec_val_dice:.4f}", flush=True)
    print(f"  Decoder pi_source={pi_source_dec:.4f}", flush=True)

    # Decoder temperature scaling
    dec_logits_val = get_decoder_logits_128(decoder, kvasir["patch_tokens"][val_idx])
    dec_logits_flat = dec_logits_val.reshape(-1).astype(np.float64)
    dec_labels_flat = kvasir_masks_128[val_idx].reshape(-1).astype(np.float64)
    temp_dec = temperature_scale(dec_logits_flat, dec_labels_flat)
    print(f"  Decoder T={temp_dec:.4f}", flush=True)

    # ── Build eval sites ────────────────────────────────────────────
    sites = [("kvasir_val", kvasir, val_idx)]
    for name in TARGET_SITES:
        sites.append((name, targets[name], np.arange(len(targets[name]["filenames"]))))

    methods = ["no_correction", "oracle", "saerens_em", "saerens_em_ts", "bbse"]

    all_lr_rows = []
    all_dec_rows = []
    cal_rows = []
    prior_rows = []
    thresh_rows = []

    for site_name, site_data, indices in sites:
        n = len(indices)
        print(f"\n  --- {site_name} ({n} images) ---", flush=True)

        # ── Cache GT masks ──────────────────────────────────────────
        gt_masks = [load_gt_mask(str(site_data["mask_paths"][idx])) for idx in indices]

        # ── Linear probe: batch predict all at once ────────────────
        X_site = site_data["patch_tokens"][indices].reshape(-1, 384)
        lr_probs_flat = clf.predict_proba(X_site)[:, 1].astype(np.float32)
        lr_logits_flat = clf.decision_function(X_site).astype(np.float64)
        lr_probs_list = [lr_probs_flat[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]
        lr_logits_list = [lr_logits_flat[i*N_PATCHES:(i+1)*N_PATCHES] for i in range(n)]

        # Upsample LR probs once (for no_correction and threshold sweep)
        lr_prob_maps = []
        for i, idx in enumerate(indices):
            ow, oh = int(site_data["orig_sizes"][idx][0]), int(site_data["orig_sizes"][idx][1])
            lr_prob_maps.append(upsample_pred_fullres(lr_probs_list[i], ow, oh))

        # ── Decoder: compute all probs & logits once ────────────────
        dec_logits_all = get_decoder_logits_128(decoder, site_data["patch_tokens"][indices])
        dec_probs_all = sigmoid(dec_logits_all)

        # Upsample decoder probs once
        dec_prob_maps = []
        for i, idx in enumerate(indices):
            ow, oh = int(site_data["orig_sizes"][idx][0]), int(site_data["orig_sizes"][idx][1])
            dec_prob_maps.append(upsample_pred_128(dec_probs_all[i], ow, oh))

        # ── Compute priors and shifts ───────────────────────────────
        pi_true = float(site_data["patch_fg_binary"][indices].reshape(-1).mean())
        lr_flat = np.concatenate(lr_probs_list)
        dec_flat = dec_probs_all.reshape(-1).astype(np.float64)

        pi_em_lr = saerens_em(lr_flat.astype(np.float64), pi_source=pi_source_lr)
        pi_em_lr_ts = saerens_em(
            sigmoid(np.concatenate(lr_logits_list) / temp_lr).astype(np.float64),
            pi_source=pi_source_lr)
        pi_bbse_lr = bbse_estimate(
            (source_probs >= 0.5).astype(int), y_val.astype(int),
            (lr_flat >= 0.5).astype(int))

        masks_128_site = load_masks_128(site_data["mask_paths"][indices])
        dec_labels_site = masks_128_site.reshape(-1).astype(np.float64)
        pi_true_dec = float(dec_labels_site.mean())

        pi_em_dec = saerens_em(dec_flat, pi_source=pi_source_dec)
        pi_em_dec_ts = saerens_em(
            sigmoid(dec_logits_all.reshape(-1).astype(np.float64) / temp_dec),
            pi_source=pi_source_dec)
        pi_bbse_dec = bbse_estimate(
            (sigmoid(dec_logits_flat) >= 0.5).astype(int),
            dec_labels_flat.astype(int),
            (dec_flat >= 0.5).astype(int))

        prior_rows.append({
            "site": site_name,
            "pi_true_patch": pi_true,
            "pi_true_128": pi_true_dec,
            "lr_em": pi_em_lr,
            "lr_em_ts": pi_em_lr_ts,
            "lr_bbse": pi_bbse_lr,
            "dec_em": pi_em_dec,
            "dec_em_ts": pi_em_dec_ts,
            "dec_bbse": pi_bbse_dec,
        })

        # ── Calibration ────────────────────────────────────────────
        lr_labels_patch = site_data["patch_fg_binary"][indices].reshape(-1).astype(np.float64)
        cal_rows.append({
            "site": site_name, "model": "linear_probe",
            "ece": expected_calibration_error(lr_flat, lr_labels_patch),
            "brier": brier_score(lr_flat, lr_labels_patch),
            "nll": nll_score(np.clip(lr_flat, 1e-10, 1-1e-10), lr_labels_patch),
        })
        cal_rows.append({
            "site": site_name, "model": "decoder",
            "ece": expected_calibration_error(dec_flat, dec_labels_site),
            "brier": brier_score(dec_flat, dec_labels_site),
            "nll": nll_score(np.clip(dec_flat, 1e-10, 1-1e-10), dec_labels_site),
        })

        # ── Evaluate each method ───────────────────────────────────
        shift_lr = {
            "oracle": logit_shift(pi_source_lr, pi_true),
            "saerens_em": logit_shift(pi_source_lr, pi_em_lr),
            "saerens_em_ts": logit_shift(pi_source_lr, pi_em_lr_ts),
            "bbse": logit_shift(pi_source_lr, pi_bbse_lr),
        }
        shift_dec = {
            "oracle": logit_shift(pi_source_dec, pi_true_dec),
            "saerens_em": logit_shift(pi_source_dec, pi_em_dec),
            "saerens_em_ts": logit_shift(pi_source_dec, pi_em_dec_ts),
            "bbse": logit_shift(pi_source_dec, pi_bbse_dec),
        }

        none_dices_lr = None
        none_dices_dec = None

        # Pre-compute ALL corrected upsampled maps
        all_lr_maps = {"no_correction": lr_prob_maps}
        all_dec_maps = {"no_correction": dec_prob_maps}
        for mtag in ["oracle", "saerens_em", "saerens_em_ts", "bbse"]:
            s_lr = shift_lr[mtag]
            s_dec = shift_dec[mtag]
            corr_lr, corr_dec = [], []
            for i, idx in enumerate(indices):
                ow, oh = int(site_data["orig_sizes"][idx][0]), int(site_data["orig_sizes"][idx][1])
                corr_lr.append(upsample_pred_fullres(
                    sigmoid(lr_logits_list[i] + s_lr).astype(np.float32), ow, oh))
                corr_dec.append(upsample_pred_128(
                    sigmoid(dec_logits_all[i] + s_dec).astype(np.float32), ow, oh))
            all_lr_maps[mtag] = corr_lr
            all_dec_maps[mtag] = corr_dec

        for method in methods:
            lr_maps = all_lr_maps[method]
            dec_maps = all_dec_maps[method]

            res_lr = evaluate_cached(lr_maps, gt_masks)
            res_dec = evaluate_cached(dec_maps, gt_masks)

            dices_lr = np.array([r["dice"] for r in res_lr])
            dices_dec = np.array([r["dice"] for r in res_dec])

            if method == "no_correction":
                none_dices_lr = dices_lr
                none_dices_dec = dices_dec

            for model_name, res, dices, none_d, out_list in [
                ("linear_probe", res_lr, dices_lr, none_dices_lr, all_lr_rows),
                ("decoder", res_dec, dices_dec, none_dices_dec, all_dec_rows),
            ]:
                ious = np.array([r["iou"] for r in res])
                precs = np.array([r["precision"] for r in res])
                recs = np.array([r["recall"] for r in res])
                pred_fgs = np.array([r["pred_fg"] for r in res])
                gt_fgs = np.array([r["gt_fg"] for r in res])
                tp_sum = sum(r["tp"] for r in res)
                fp_sum = sum(r["fp"] for r in res)
                fn_sum = sum(r["fn"] for r in res)
                pooled_d = 2 * tp_sum / max(2 * tp_sum + fp_sum + fn_sum, 1)
                fp_empty = sum(1 for r in res if r["fp_on_empty"])
                n_empty = sum(1 for r in res if r["gt_fg"] == 0)

                deltas = dices - none_d if none_d is not None else np.zeros_like(dices)
                if method != "no_correction" and none_d is not None:
                    ci = bootstrap_ci(deltas)
                else:
                    ci = (0.0, 0.0)

                out_list.append({
                    "site": site_name, "method": method, "n": n,
                    "dice_mean": float(dices.mean()),
                    "dice_std": float(dices.std()),
                    "dice_pooled": float(pooled_d),
                    "iou_mean": float(ious.mean()),
                    "prec_mean": float(precs.mean()),
                    "rec_mean": float(recs.mean()),
                    "pred_fg_mean": float(pred_fgs.mean()),
                    "gt_fg_mean": float(gt_fgs.mean()),
                    "fp_empty": fp_empty, "n_empty": n_empty,
                    "delta": float(deltas.mean()),
                    "delta_lo": ci[0], "delta_hi": ci[1],
                })

        # ── Threshold sweep (Step 6) — reuse pre-computed maps ───────
        sweep_tags = ["no_correction", "oracle", "saerens_em"]
        for thresh in np.arange(0.1, 0.95, 0.1):
            t = float(thresh)
            for mtag in sweep_tags:
                d_lr = [dice_score((m >= t).astype(np.uint8), g)
                        for m, g in zip(all_lr_maps[mtag], gt_masks)]
                d_dec = [dice_score((m >= t).astype(np.uint8), g)
                         for m, g in zip(all_dec_maps[mtag], gt_masks)]
                thresh_rows.append({"site": site_name, "model": "linear_probe",
                                    "method": mtag, "threshold": t,
                                    "dice_mean": float(np.mean(d_lr))})
                thresh_rows.append({"site": site_name, "model": "decoder",
                                    "method": mtag, "threshold": t,
                                    "dice_mean": float(np.mean(d_dec))})

        del gt_masks, lr_prob_maps, dec_prob_maps, all_lr_maps, all_dec_maps

    # ═══════════════════════════════════════════════════════════════
    # PRINT ALL TABLES
    # ═══════════════════════════════════════════════════════════════

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 25)

    # ── Table 1: Linear probe ───────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("TABLE 1: LINEAR PROBE — corrected results", flush=True)
    print("=" * 100, flush=True)
    df_lr = pd.DataFrame(all_lr_rows)
    cols = ["site", "method", "n", "dice_mean", "dice_pooled", "prec_mean", "rec_mean",
            "pred_fg_mean", "gt_fg_mean", "fp_empty", "n_empty", "delta", "delta_lo", "delta_hi"]
    print(df_lr[cols].to_string(index=False), flush=True)

    # ── Table 2: Decoder ────────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("TABLE 2: DECODER (corrected reshape) — corrected results", flush=True)
    print("=" * 100, flush=True)
    df_dec = pd.DataFrame(all_dec_rows)
    print(df_dec[cols].to_string(index=False), flush=True)

    # ── Table 3: Calibration ────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("TABLE 3: CALIBRATION (patch/pixel-level)", flush=True)
    print("=" * 100, flush=True)
    df_cal = pd.DataFrame(cal_rows)
    print(df_cal.to_string(index=False), flush=True)

    # ── Table 4: Prior estimates ────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("TABLE 4: PRIOR ESTIMATES — estimated vs true", flush=True)
    print(f"  LR temperature: {temp_lr:.4f}   Decoder temperature: {temp_dec:.4f}", flush=True)
    print("=" * 100, flush=True)
    df_prior = pd.DataFrame(prior_rows)
    print(df_prior.to_string(index=False), flush=True)

    # ── Table 5: Threshold sweep (best per site) ────────────────────
    print("\n" + "=" * 100, flush=True)
    print("TABLE 5: BEST THRESHOLD per site/method (upper bound — uses target labels)", flush=True)
    print("=" * 100, flush=True)
    df_thresh = pd.DataFrame(thresh_rows)
    best = df_thresh.loc[df_thresh.groupby(["site", "model", "method"])["dice_mean"].idxmax()]
    best = best[["site", "model", "method", "threshold", "dice_mean"]]
    print(best.to_string(index=False), flush=True)

    # Full sweep table
    print("\n" + "=" * 100, flush=True)
    print("TABLE 5b: FULL THRESHOLD SWEEP (linear probe)", flush=True)
    print("=" * 100, flush=True)
    lr_thresh = df_thresh[df_thresh["model"] == "linear_probe"]
    pivot = lr_thresh.pivot_table(index=["site", "method"], columns="threshold", values="dice_mean")
    print(pivot.round(4).to_string(), flush=True)

    # ── Save CSVs ───────────────────────────────────────────────────
    df_lr.to_csv(OUT_DIR / "linear_probe_results.csv", index=False)
    df_dec.to_csv(OUT_DIR / "decoder_results.csv", index=False)
    df_cal.to_csv(OUT_DIR / "calibration.csv", index=False)
    df_prior.to_csv(OUT_DIR / "prior_estimates.csv", index=False)
    df_thresh.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)

    # ── Mark old tables ─────────────────────────────────────────────
    for d in [ROOT / "outputs" / "03_main_experiments" / "oracle",
              ROOT / "outputs" / "03_main_experiments" / "decoder"]:
        marker = d / "SUPERSEDED.txt"
        if d.exists() and not marker.exists():
            marker.write_text(
                "Results INVALID: float64→PIL bug + decoder reshape bug.\n"
                "See outputs/04_corrected/ for corrected results.\n")
            print(f"  Marked {d} as superseded", flush=True)

    print(f"\nAll CSVs saved to {OUT_DIR}/", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
