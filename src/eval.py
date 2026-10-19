"""Shared evaluation utilities for polyp segmentation experiments.

All metric computation, upsampling, thresholding, calibration,
and prior estimation code lives here. Every experiment script
must import from this module — no duplicate metric code.
"""

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar

GRID = 16
N_PATCHES = GRID * GRID


# ── Numerics ────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Mask / upsample ────────────────────────────────────────────────

def load_gt_mask(path, threshold=128):
    return (np.array(Image.open(str(path)).convert("L")) >= threshold).astype(np.uint8)


def upsample_probs(prob_16x16, target_w, target_h):
    """Upsample a 16x16 probability grid to (target_w, target_h).

    IMPORTANT: input MUST be float32. PIL mode 'F' silently
    misinterprets float64 bytes, producing garbage.
    """
    arr = np.asarray(prob_16x16, dtype=np.float32)
    img = Image.fromarray(arr, mode="F")
    return np.array(img.resize((int(target_w), int(target_h)), Image.BILINEAR))


# ── Segmentation metrics ───────────────────────────────────────────

def dice_score(pred, gt):
    pred = np.asarray(pred, dtype=bool).ravel()
    gt = np.asarray(gt, dtype=bool).ravel()
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def iou_score(pred, gt):
    pred = np.asarray(pred, dtype=bool).ravel()
    gt = np.asarray(gt, dtype=bool).ravel()
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def pixel_metrics(prob_patches_256, mask_path, orig_w, orig_h, threshold=0.5):
    """Full pipeline: reshape 256 probs → 16x16, upsample, threshold, metrics."""
    prob_grid = np.asarray(prob_patches_256, dtype=np.float32).reshape(GRID, GRID)
    prob_up = upsample_probs(prob_grid, orig_w, orig_h)
    pred = (prob_up >= threshold).astype(np.uint8)
    gt = load_gt_mask(mask_path)

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())

    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "pred_fg": int(pred_b.sum()),
        "gt_fg": int(gt_b.sum()),
        "fp_on_empty": bool(gt_b.sum() == 0 and pred_b.sum() > 0),
        "tp": tp, "fp": fp, "fn": fn,
    }


def pixel_metrics_128(prob_128, mask_path, orig_w, orig_h, threshold=0.5):
    """Same pipeline but input is 128x128 (decoder output)."""
    arr = np.asarray(prob_128, dtype=np.float32)
    img = Image.fromarray(arr, mode="F")
    prob_up = np.array(img.resize((int(orig_w), int(orig_h)), Image.BILINEAR))
    pred = (prob_up >= threshold).astype(np.uint8)
    gt = load_gt_mask(mask_path)

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())

    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "pred_fg": int(pred_b.sum()),
        "gt_fg": int(gt_b.sum()),
        "fp_on_empty": bool(gt_b.sum() == 0 and pred_b.sum() > 0),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Calibration metrics ────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        ece += n * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece / len(probs))


def brier_score(probs, labels):
    return float(((probs - labels.astype(float)) ** 2).mean())


def nll_score(probs, labels):
    eps = 1e-10
    p = np.clip(probs, eps, 1 - eps)
    y = labels.astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ── Prior correction ───────────────────────────────────────────────

def logit_shift(pi_source, pi_target):
    eps = 1e-7
    ps = np.clip(float(pi_source), eps, 1 - eps)
    pt = np.clip(float(pi_target), eps, 1 - eps)
    return float(np.log(pt / (1 - pt)) - np.log(ps / (1 - ps)))


def saerens_em(probs, pi_source=None, max_iter=100, tol=1e-6):
    """Saerens et al. EM for target prior estimation.

    pi_source: class prior the classifier was trained under.
               If None, uses mean of predicted probs (original formulation).
    Returns estimated target prior.
    """
    pi_s = float(probs.mean()) if pi_source is None else float(pi_source)
    pi_t = float(probs.mean())
    p = probs.copy().astype(np.float64)

    for _ in range(max_iter):
        r_pos = pi_t / (pi_s + 1e-10)
        r_neg = (1 - pi_t) / (1 - pi_s + 1e-10)
        num = p * r_pos
        q = num / (num + (1 - p) * r_neg + 1e-10)
        pi_new = float(q.mean())
        if abs(pi_new - pi_t) < tol:
            break
        pi_t = pi_new
    return pi_t


def bbse_estimate(preds_source, labels_source, preds_target):
    """BBSE (Lipton et al. 2018) using hard predictions + confusion matrix."""
    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C[i, j] = ((preds_source == i) & (labels_source == j)).sum()
    col_sums = C.sum(axis=0)
    col_sums[col_sums == 0] = 1
    C = C / col_sums

    mu_t = np.array([(preds_target == 0).mean(), (preds_target == 1).mean()])
    try:
        w = np.linalg.solve(C, mu_t)
        w = np.clip(w, 1e-8, 1.0)
        w = w / w.sum()
        return float(w[1])
    except np.linalg.LinAlgError:
        return float(preds_target.mean())


def temperature_scale(logits, labels):
    """Find optimal temperature T on validation NLL via bounded minimization."""
    labels = labels.astype(float)

    def nll_at_T(T):
        p = sigmoid(logits / T)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()

    result = minimize_scalar(nll_at_T, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def bias_corrected_temperature_scale(logits, labels):
    """Bias-corrected temperature scaling: p = sigmoid(logit/T + c).
    Returns (T, c). The bias c absorbs prior mismatch from class reweighting."""
    from scipy.optimize import minimize as _minimize
    labels = labels.astype(float)

    def nll(params):
        T, c = params
        if T < 0.01:
            return 1e6
        p = sigmoid(logits / T + c)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

    result = _minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                       options={"maxiter": 5000})
    return float(result.x[0]), float(result.x[1])


# ── Bootstrap ──────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.array([rng.choice(values, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


# ── Data loading ───────────────────────────────────────────────────

def load_site_features(feat_dir, name):
    d = np.load(feat_dir / f"{name}.npz")
    return {
        "patch_tokens": d["patch_tokens"].astype(np.float32),
        "patch_fg_binary": d["patch_fg_binary"],
        "patch_fg_fraction": d["patch_fg_fraction"].astype(np.float32),
        "filenames": d["filenames"],
        "mask_paths": d["mask_paths"],
        "orig_sizes": d["orig_sizes"],
    }
