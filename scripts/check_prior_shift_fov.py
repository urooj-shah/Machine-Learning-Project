"""Check whether the foreground-fraction gap between sites survives
after normalizing by the visible field-of-view (FOV) area.

Camera borders/overlays differ by site and inflate total pixel counts
differently. This script estimates the FOV from the image itself, then
computes fg_pixels_inside_FOV / FOV_pixels.

Matching rules (verified by inspection):
  PolypGen C1-C6: mask "{stem}_mask.jpg" -> image "{stem}.jpg"
  Kvasir-SEG:     same filename in images/ and masks/
"""

import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_FOV = ROOT / "outputs" / "01_sanity_check" / "fov"
OUT_CHECK = ROOT / "outputs" / "01_sanity_check" / "fov_check"

POLYPGEN = RAW / "PolypGen2021_MultiCenterData_v3"
KVASIR = RAW / "kvasir-seg" / "Kvasir-SEG" / "Kvasir-SEG"

MASK_THRESH = 128
FOV_THRESH = 12

random.seed(42)


# ── Data layout ──────────────────────────────────────────────────────

def get_pairs(center_name):
    """Return list of (image_path, mask_path) for a center."""
    if center_name == "Kvasir":
        img_dir = KVASIR / "images"
        mask_dir = KVASIR / "masks"
        imgs = set(os.listdir(img_dir))
        masks = set(os.listdir(mask_dir))
        common = sorted(imgs & masks)
        return [(img_dir / f, mask_dir / f) for f in common]
    else:
        c = center_name  # e.g. "C1"
        img_dir = POLYPGEN / f"data_{c}" / f"images_{c}"
        mask_dir = POLYPGEN / f"data_{c}" / f"masks_{c}"
        imgs = set(os.listdir(img_dir))
        pairs = []
        for mf in sorted(os.listdir(mask_dir)):
            stem = mf.replace("_mask.jpg", ".jpg")
            if stem in imgs:
                pairs.append((img_dir / stem, mask_dir / mf))
        return pairs


# ── FOV estimation ───────────────────────────────────────────────────

def estimate_fov(img_bgr):
    """Estimate field-of-view mask from an endoscopy image.

    Grayscale > threshold, keep largest connected component, fill holes,
    morphological close.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, FOV_THRESH, 255, cv2.THRESH_BINARY)

    n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return np.ones(gray.shape, dtype=np.uint8) * 255

    # Largest component (skip label 0 = background)
    areas = comp_stats[1:, cv2.CC_STAT_AREA]
    largest = np.argmax(areas) + 1
    fov = np.zeros(gray.shape, dtype=np.uint8)
    fov[labels == largest] = 255

    # Fill holes
    contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fov, contours, -1, 255, cv2.FILLED)

    # Morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)

    return fov


# ── Per-image computation ────────────────────────────────────────────

def compute_fractions(pairs):
    """Return raw and FOV-adjusted fg fractions for positive masks only."""
    raw_ratios = []
    fov_ratios = []
    fov_pixels_list = []
    used_pairs = []

    for img_path, mask_path in tqdm(pairs, leave=False):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        fg = (mask >= MASK_THRESH).astype(np.uint8)

        if fg.sum() == 0:
            continue

        img = cv2.imread(str(img_path))
        fov = estimate_fov(img)
        fov_bin = (fov > 0).astype(np.uint8)
        fov_area = fov_bin.sum()

        if fov_area == 0:
            continue

        fg_in_fov = (fg & fov_bin).sum()
        raw_ratios.append(fg.sum() / fg.size)
        fov_ratios.append(fg_in_fov / fov_area)
        fov_pixels_list.append(fov_area)
        used_pairs.append((img_path, mask_path))

    return {
        "raw": np.array(raw_ratios),
        "fov": np.array(fov_ratios),
        "fov_pixels": np.array(fov_pixels_list),
        "pairs": used_pairs,
    }


# ── Overlay grid ─────────────────────────────────────────────────────

def save_overlay_grid(center_name, results, n=5):
    """Save a grid of FOV outlines on random images for visual check."""
    pairs = results["pairs"]
    if len(pairs) == 0:
        return
    indices = random.sample(range(len(pairs)), min(n, len(pairs)))

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_path, _ = pairs[idx]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fov = estimate_fov(img)
        contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img_rgb.copy()
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 3)

        ax.imshow(overlay)
        ax.set_title(Path(img_path).name, fontsize=7)
        ax.axis("off")

    fig.suptitle(f"{center_name} — FOV outline (red)", fontsize=12)
    fig.tight_layout()
    OUT_CHECK.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_CHECK / f"fov_overlay_{center_name}.png", dpi=150)
    plt.close()


# ── Bootstrap CI for ratio ───────────────────────────────────────────

def bootstrap_ratio(a, b, n_boot=5000, ci=0.95):
    """Bootstrap 95% CI for median(a) / median(b)."""
    rng = np.random.default_rng(42)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        ratios[i] = np.median(sa) / np.median(sb)
    lo = np.percentile(ratios, (1 - ci) / 2 * 100)
    hi = np.percentile(ratios, (1 + ci) / 2 * 100)
    return np.median(ratios), lo, hi


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUT_FOV.mkdir(parents=True, exist_ok=True)
    OUT_CHECK.mkdir(parents=True, exist_ok=True)

    center_names = ["C1", "C2", "C3", "C4", "C5", "C6", "Kvasir"]
    all_results = {}

    for name in center_names:
        print(f"Processing {name} ...")
        pairs = get_pairs(name)
        results = compute_fractions(pairs)
        save_overlay_grid(name, results)
        all_results[name] = results
        print(f"  {len(results['fov'])} positive images")

    # ── Summary table ────────────────────────────────────────────────
    kvasir_fov = all_results["Kvasir"]["fov"]
    kvasir_raw = all_results["Kvasir"]["raw"]

    rows = []
    print()
    header = (
        f"{'Center':<8} {'N':>4}  "
        f"{'Raw Med%':>8} {'FOV Med%':>8}  "
        f"{'Ratio vs Kvasir (raw)':>22} {'Ratio vs Kvasir (FOV)':>22}"
    )
    print(header)
    print("-" * len(header))

    for name in center_names:
        r = all_results[name]
        n = len(r["fov"])
        raw_med = np.median(r["raw"]) * 100
        fov_med = np.median(r["fov"]) * 100

        if name != "Kvasir":
            raw_ratio, raw_lo, raw_hi = bootstrap_ratio(kvasir_raw, r["raw"])
            fov_ratio, fov_lo, fov_hi = bootstrap_ratio(kvasir_fov, r["fov"])
            raw_str = f"{raw_ratio:.2f} [{raw_lo:.2f}-{raw_hi:.2f}]"
            fov_str = f"{fov_ratio:.2f} [{fov_lo:.2f}-{fov_hi:.2f}]"
        else:
            raw_str = "1.00 (ref)"
            fov_str = "1.00 (ref)"

        print(f"{name:<8} {n:>4}  {raw_med:>8.2f} {fov_med:>8.2f}  {raw_str:>22} {fov_str:>22}")
        rows.append({
            "center": name, "n": n,
            "raw_median_pct": round(raw_med, 3),
            "fov_median_pct": round(fov_med, 3),
        })

    # ── Statistical tests (FOV-adjusted, positive only) ──────────────
    print("\n--- Statistical tests (FOV-adjusted, positive only) ---")
    kw_stat, kw_p = stats.kruskal(*(all_results[n]["fov"] for n in center_names))
    print(f"Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.2e}")

    print("Pairwise Mann-Whitney U (two-sided):")
    pairs_to_test = [("Kvasir", "C4"), ("Kvasir", "C1"), ("C1", "C4"), ("C1", "C5")]
    for a, b in pairs_to_test:
        u, p = stats.mannwhitneyu(all_results[a]["fov"], all_results[b]["fov"], alternative="two-sided")
        print(f"  {a:>6} vs {b:<6}  U={u:>10.0f}  p={p:.2e}")

    # ── Save CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = OUT_FOV / "prior_shift_fov_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    # ── Box plot: side by side raw vs FOV ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd", "#dd8452"]
    labels = [f"{n}\n(n={len(all_results[n]['fov'])})" for n in center_names]

    for ax, key, title in [
        (ax1, "raw", "Raw (fg / total pixels)"),
        (ax2, "fov", "FOV-adjusted (fg∩FOV / FOV pixels)"),
    ]:
        data = [all_results[n][key] * 100 for n in center_names]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel("Foreground fraction (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prior shift: raw vs FOV-adjusted (positive images only)", fontsize=13)
    fig.tight_layout()
    plot_path = OUT_FOV / "prior_shift_raw_vs_fov.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Box plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
