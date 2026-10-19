"""Sanity check: does polyp foreground fraction differ across centers?

Loads masks from PolypGen (C1-C6) and Kvasir-SEG, computes per-image
foreground ratio, and reports per-center statistics + a box plot.

Masks are JPEG (lossy), so we threshold at 128 to binarize.
Multi-channel masks are converted to grayscale first.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

POLYPGEN = RAW / "PolypGen2021_MultiCenterData_v3"
KVASIR = RAW / "kvasir-seg" / "Kvasir-SEG" / "Kvasir-SEG"

THRESHOLD = 128


def fg_ratio(mask_path: str) -> float:
    """Return foreground pixel fraction for a single mask."""
    m = np.array(Image.open(mask_path).convert("L"))
    binary = (m >= THRESHOLD).astype(np.uint8)
    return binary.sum() / binary.size


def collect_center(name: str, mask_dir: Path) -> dict:
    """Compute fg ratios for all masks in a directory."""
    files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith((".jpg", ".png", ".bmp")))
    ratios = []
    empty = 0
    for f in tqdm(files, desc=name, leave=False):
        r = fg_ratio(mask_dir / f)
        ratios.append(r)
        if r == 0.0:
            empty += 1
    return {
        "name": name,
        "n": len(ratios),
        "empty": empty,
        "ratios": np.array(ratios),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    centers = []
    for c in range(1, 7):
        mask_dir = POLYPGEN / f"data_C{c}" / f"masks_C{c}"
        centers.append(collect_center(f"C{c}", mask_dir))

    centers.append(collect_center("Kvasir", KVASIR / "masks"))

    # --- Print summary tables ---
    for label, filter_empty in [("ALL images", False), ("POSITIVE-only (fg > 0)", True)]:
        print(f"\n=== {label} ===")
        print(f"{'Center':<10} {'N':>5} {'Empty':>6} {'Mean%':>7} {'Median%':>8} {'Std%':>6} {'Min%':>6} {'Max%':>6}")
        print("-" * 62)
        for c in centers:
            r = c["ratios"]
            if filter_empty:
                r = r[r > 0]
            r_pct = r * 100
            empty = (c["ratios"] == 0).sum()
            print(
                f"{c['name']:<10} {len(r):>5} {empty:>6} "
                f"{r_pct.mean():>7.2f} {np.median(r_pct):>8.2f} {r_pct.std():>6.2f} "
                f"{r_pct.min():>6.2f} {r_pct.max():>6.2f}"
            )

        # --- Statistical tests ---
        subset = []
        for c in centers:
            r = c["ratios"]
            if filter_empty:
                r = r[r > 0]
            subset.append(r)

        print(f"\n  Kruskal-Wallis:", end=" ")
        kw_stat, kw_p = stats.kruskal(*subset)
        print(f"H={kw_stat:.2f}, p={kw_p:.2e}")

        print("  Pairwise Mann-Whitney U (two-sided):")
        names = [c["name"] for c in centers]
        pairs = [("Kvasir", "C4"), ("Kvasir", "C1"), ("C1", "C4"), ("C1", "C5")]
        for a, b in pairs:
            ra = subset[names.index(a)]
            rb = subset[names.index(b)]
            u_stat, u_p = stats.mannwhitneyu(ra, rb, alternative="two-sided")
            print(f"    {a:>6} vs {b:<6}  U={u_stat:>10.0f}  p={u_p:.2e}")

    # --- Box plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [c["ratios"] * 100 for c in centers]
    labels = [f"{c['name']}\n(n={c['n']})" for c in centers]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd", "#dd8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Foreground fraction (%)")
    ax.set_title("Polyp foreground fraction by center / dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT / "prior_shift_boxplot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nBox plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
