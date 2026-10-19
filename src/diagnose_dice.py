"""Diagnose why patch-level classifier Dice is low.

1. Quantization ceiling (patch labels upsampled vs true mask)
2. Overlay visualizations (pred vs GT patch grids)
3. Train/val patch-level AUC, F1
4. Empty-mask handling
5. Mean per-image Dice vs pooled Dice
6. Feature standardization / class weight / regularization ablation
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "03_main_experiments" / "oracle" / "debug"

GRID = 16
N_PATCHES = GRID * GRID
SEED = 42


def load_site(name):
    d = np.load(FEAT_DIR / f"{name}.npz")
    return {
        "patch_tokens": d["patch_tokens"].astype(np.float32),
        "patch_fg_binary": d["patch_fg_binary"],
        "patch_fg_fraction": d["patch_fg_fraction"].astype(np.float32),
        "filenames": d["filenames"],
        "mask_paths": d["mask_paths"],
        "orig_sizes": d["orig_sizes"],
    }


def upsample_patches_to_mask(patch_grid_16x16, orig_w, orig_h):
    """Upsample a 16x16 float grid to original mask resolution."""
    img = Image.fromarray(patch_grid_16x16.astype(np.float32), mode="F")
    return np.array(img.resize((orig_w, orig_h), Image.BILINEAR))


def dice(pred, gt):
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2 * inter / denom


def load_gt_mask(mask_path):
    return (np.array(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)


# ── 1. Quantization ceiling ─────────────────────────────────────────

def quantization_ceiling(data, indices, label):
    """Dice of upsampled GT patch labels vs original masks."""
    dices = []
    for idx in indices:
        gt_patches = data["patch_fg_binary"][idx].reshape(GRID, GRID).astype(np.float32)
        orig_w, orig_h = data["orig_sizes"][idx]
        upsampled = upsample_patches_to_mask(gt_patches, int(orig_w), int(orig_h))
        pred = (upsampled >= 0.5).astype(np.uint8)
        gt = load_gt_mask(str(data["mask_paths"][idx]))
        d = dice(pred, gt)
        dices.append(d)

    pos_mask = data["patch_fg_binary"][indices].sum(axis=1) > 0
    pos_indices = indices[pos_mask]
    dices_arr = np.array(dices)
    pos_dices = dices_arr[pos_mask]

    print(f"  {label}:")
    print(f"    All:      n={len(indices):>4}  mean_dice={dices_arr.mean():.4f}  median={np.median(dices_arr):.4f}")
    print(f"    Pos only: n={len(pos_indices):>4}  mean_dice={pos_dices.mean():.4f}  median={np.median(pos_dices):.4f}")
    return dices_arr


# ── 2. Overlay visualizations ────────────────────────────────────────

def save_overlays(data, indices, clf, scaler, out_dir):
    """12 random val images: predicted vs GT patch grid overlaid on image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    sample = random.sample(list(indices), min(12, len(indices)))

    n_rows = 12
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))

    for i, idx in enumerate(sample):
        row = i
        col_gt = 0
        col_pred = 1

        # Load original image (resized to 224 for display)
        img_dir = Path(str(data["mask_paths"][idx])).parent.parent
        fname = str(data["filenames"][idx])
        # For Kvasir: mask and image have same filename
        # Reconstruct image path from mask path
        mask_p = str(data["mask_paths"][idx])
        img_p = mask_p.replace("/masks/", "/images/").replace("_mask.", ".")
        if not os.path.exists(img_p):
            img_p = mask_p.replace("/masks/", "/images/")
        img = Image.open(img_p).convert("RGB").resize((224, 224))
        img_arr = np.array(img)

        # GT patch grid
        gt_patches = data["patch_fg_binary"][idx].reshape(GRID, GRID)

        # Predicted patch grid
        X = data["patch_tokens"][idx].reshape(N_PATCHES, -1)
        if scaler is not None:
            X = scaler.transform(X)
        probs = clf.predict_proba(X)[:, 1].reshape(GRID, GRID)
        pred_patches = (probs >= 0.5).astype(np.uint8)

        # Overlay GT on image
        gt_up = np.array(Image.fromarray(gt_patches.astype(np.float32) * 255, mode="F").resize((224, 224), Image.NEAREST))
        overlay_gt = img_arr.copy()
        mask_region = gt_up >= 128
        overlay_gt[mask_region] = (overlay_gt[mask_region] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Overlay pred on image
        pred_up = np.array(Image.fromarray(probs.astype(np.float32), mode="F").resize((224, 224), Image.NEAREST))
        overlay_pred = img_arr.copy()
        pred_region = pred_up >= 0.5
        overlay_pred[pred_region] = (overlay_pred[pred_region] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        # Plot
        axes[row, col_gt].imshow(overlay_gt)
        axes[row, col_gt].set_title(f"GT (green) — {fname[:20]}", fontsize=8)
        axes[row, col_gt].axis("off")

        axes[row, col_pred].imshow(overlay_pred)
        d = dice(pred_patches.reshape(-1), gt_patches.reshape(-1))
        axes[row, col_pred].set_title(f"Pred (red) — patch_dice={d:.2f}", fontsize=8)
        axes[row, col_pred].axis("off")

    fig.suptitle("Patch-level GT (green) vs Predicted (red) — 12 Kvasir val images", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "patch_overlays.png", dpi=150)
    plt.close()
    print(f"  Overlays saved to {out_dir / 'patch_overlays.png'}")


# ── 3-6. Train classifier and evaluate ──────────────────────────────

def train_and_evaluate(data, train_idx, val_idx, C=1.0, class_weight=None,
                       standardize=False, label=""):
    """Train LR, report patch-level and pixel-level metrics."""
    X_train = data["patch_tokens"][train_idx].reshape(-1, data["patch_tokens"].shape[-1])
    y_train = data["patch_fg_binary"][train_idx].reshape(-1)
    X_val = data["patch_tokens"][val_idx].reshape(-1, data["patch_tokens"].shape[-1])
    y_val = data["patch_fg_binary"][val_idx].reshape(-1)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                             C=C, class_weight=class_weight)
    clf.fit(X_train, y_train)

    # Patch-level metrics
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        acc = (preds == y).mean()
        print(f"    {split_name}: AUC={auc:.4f}  F1={f1:.4f}  acc={acc:.4f}  pos_rate={y.mean():.4f}")

    # Pixel-level Dice (val)
    all_dices = []
    pos_dices = []
    empty_fp = []  # false positive rate on empty masks

    # Also compute pooled dice
    pooled_inter = 0
    pooled_denom = 0

    for idx in val_idx:
        X_img = data["patch_tokens"][idx].reshape(N_PATCHES, -1)
        if scaler is not None:
            X_img = scaler.transform(X_img)
        probs_img = clf.predict_proba(X_img)[:, 1].reshape(GRID, GRID)

        orig_w, orig_h = data["orig_sizes"][idx]
        upsampled = upsample_patches_to_mask(probs_img, int(orig_w), int(orig_h))
        pred = (upsampled >= 0.5).astype(np.uint8)
        gt = load_gt_mask(str(data["mask_paths"][idx]))

        inter = (pred & gt).sum()
        denom = pred.sum() + gt.sum()

        if gt.sum() == 0:
            # Empty mask image
            if pred.sum() == 0:
                d = 1.0  # both empty -> perfect
            else:
                d = 0.0  # false positive on empty mask
            empty_fp.append(pred.sum() > 0)
        else:
            d = 2 * inter / denom if denom > 0 else 1.0
            pos_dices.append(d)

        all_dices.append(d)
        pooled_inter += inter
        pooled_denom += denom

    all_dices = np.array(all_dices)
    pos_dices = np.array(pos_dices)
    pooled_dice = 2 * pooled_inter / pooled_denom if pooled_denom > 0 else 0

    n_empty = len(empty_fp)
    fp_rate = np.mean(empty_fp) if empty_fp else 0

    print(f"    Pixel Dice (val, all):       n={len(all_dices):>4}  mean={all_dices.mean():.4f}  std={all_dices.std():.4f}")
    print(f"    Pixel Dice (val, pos only):  n={len(pos_dices):>4}  mean={pos_dices.mean():.4f}  std={pos_dices.std():.4f}")
    print(f"    Pixel Dice (val, pooled):    {pooled_dice:.4f}")
    print(f"    Empty masks in val: {n_empty}, FP rate on empty: {fp_rate:.4f}")

    return clf, scaler, all_dices


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Kvasir ...")
    kvasir = load_site("kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    filenames = list(kvasir["filenames"])
    train_idx = np.array([filenames.index(fn) for fn in split["train"]])
    val_idx = np.array([filenames.index(fn) for fn in split["val"]])

    # ── 1. Quantization ceiling ──────────────────────────────────────
    print("\n=== 1. Quantization ceiling (GT patches upsampled -> Dice vs true mask) ===")
    quantization_ceiling(kvasir, val_idx, "Kvasir val")
    quantization_ceiling(kvasir, train_idx, "Kvasir train")

    for name in ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                 "polypgen_C4", "polypgen_C5", "polypgen_C6"]:
        site = load_site(name)
        all_idx = np.arange(len(site["filenames"]))
        quantization_ceiling(site, all_idx, name)

    # ── 3. Default classifier ────────────────────────────────────────
    print("\n=== 3. Default classifier (C=1, balanced, no scaling) ===")
    clf_default, scaler_default, _ = train_and_evaluate(
        kvasir, train_idx, val_idx,
        C=1.0, class_weight="balanced", standardize=False, label="default"
    )

    # ── 2. Overlays ──────────────────────────────────────────────────
    print("\n=== 2. Overlay visualizations ===")
    save_overlays(kvasir, val_idx, clf_default, scaler_default, OUT_DIR)

    # ── 6. Ablations ─────────────────────────────────────────────────
    print("\n=== 6. Ablation: standardized features ===")
    train_and_evaluate(kvasir, train_idx, val_idx,
                       C=1.0, class_weight="balanced", standardize=True)

    print("\n=== 6. Ablation: no class weighting ===")
    train_and_evaluate(kvasir, train_idx, val_idx,
                       C=1.0, class_weight=None, standardize=False)

    print("\n=== 6. Ablation: lower regularization (C=10) ===")
    train_and_evaluate(kvasir, train_idx, val_idx,
                       C=10.0, class_weight="balanced", standardize=False)

    print("\n=== 6. Ablation: much lower regularization (C=100) ===")
    train_and_evaluate(kvasir, train_idx, val_idx,
                       C=100.0, class_weight="balanced", standardize=False)

    print("\n=== 6. Ablation: standardized + C=10 + balanced ===")
    train_and_evaluate(kvasir, train_idx, val_idx,
                       C=10.0, class_weight="balanced", standardize=True)

    # ── 4. How empty-mask images are scored ──────────────────────────
    print("\n=== 4. Empty-mask scoring rule ===")
    print("  If gt.sum()==0 and pred.sum()==0: Dice = 1.0 (both empty, correct)")
    print("  If gt.sum()==0 and pred.sum()>0:  Dice = 0.0 (false positive)")
    print("  If gt.sum()>0:  Dice = 2*inter/(pred+gt) as usual")
    print("  Kvasir has 0 or 1 empty masks in val, so this is not the issue.")

    print("\nDone. Check overlays in", OUT_DIR)


if __name__ == "__main__":
    main()
