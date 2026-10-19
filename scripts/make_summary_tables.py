#!/usr/bin/env python3
"""Generate summary tables from saved experiment CSVs.

Reads:
  outputs/08_controls/ctrl_448_comparison.csv
  outputs/08_controls/ctrl_fewlabel.csv
  outputs/11_ablations/ablation_raw.csv

Writes:
  outputs/12_summary/tables.md

Every number in PROJECT_SUMMARY.md should trace back to this script.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "12_summary"
OUT.mkdir(parents=True, exist_ok=True)

SITE_ORDER = [
    "polypgen_C1", "polypgen_C2", "polypgen_C3",
    "polypgen_C4", "polypgen_C5", "polypgen_C6",
]
SITE_SHORT = {s: s.replace("polypgen_", "") for s in SITE_ORDER}

VARIANT_ORDER = [
    "source_448", "ensemble_224_448",
    "self_train_uniform", "ours_agree_224_448", "agree_448_448",
]
VARIANT_NICE = {
    "source_448": "Source 448",
    "ensemble_224_448": "Ensemble",
    "self_train_uniform": "Uniform PL",
    "ours_agree_224_448": "Agree 224-448",
    "agree_448_448": "Agree 448-448",
}

lines = []

def add(s=""):
    lines.append(s)


def fmt(v, n=3):
    return f"{v:.{n}f}"


def fmtpm(mean, std, n=3):
    return f"{mean:.{n}f} +/- {std:.{n}f}"


# ── Table 1: Resolution comparison ──────────────────────────────────────────
res_df = pd.read_csv(ROOT / "outputs/08_controls/ctrl_448_comparison.csv")

add("## Table 1: Resolution Comparison (224 vs 448)")
add("")
add("Positives-only Dice, mean over 5 seeds (seeds 42-46). Features: DINOv2 ViT-S/14.")
add("")
add("| Site | Dice 224 | Dice 448 | Delta |")
add("|---|---|---|---|")
for _, r in res_df.iterrows():
    site = r["site"]
    d224 = r["dice_224_mean"]
    d448 = r["dice_448_mean"]
    delta = d448 - d224
    add(f"| {site} | {d224:.3f} | {d448:.3f} | {delta:+.3f} |")
add("")

# ── Table 2: Ablation — absolute Dice ───────────────────────────────────────
abl_df = pd.read_csv(ROOT / "outputs/11_ablations/ablation_raw.csv")

add("## Table 2: Ablation — Positives-Only Dice (5 seeds, original mask resolution)")
add("")
add("Mean +/- std over seeds 42-46. Evaluation at original GT mask resolution via bilinear upsample.")
add("")

header = "| Site |"
for v in VARIANT_ORDER:
    header += f" {VARIANT_NICE[v]} |"
add(header)
add("|---|" + "---|" * len(VARIANT_ORDER))

means_all = {v: [] for v in VARIANT_ORDER}

for site in SITE_ORDER:
    row = f"| {SITE_SHORT[site]} |"
    for v in VARIANT_ORDER:
        sub = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == v)]
        m = sub["dice_pos"].mean()
        s = sub["dice_pos"].std()
        means_all[v].append(m)
        row += f" {fmtpm(m, s)} |"
    add(row)

row_mean = "| **Mean** |"
for v in VARIANT_ORDER:
    row_mean += f" {fmt(np.mean(means_all[v]))} |"
add(row_mean)
add("")

# ── Table 3: Ablation — delta vs source_448 ────────────────────────────────
add("## Table 3: Ablation — Delta vs Source 448")
add("")
add("Win = positive mean delta. Bold = best variant per site.")
add("")

delta_variants = [v for v in VARIANT_ORDER if v != "source_448"]
header = "| Site |"
for v in delta_variants:
    header += f" {VARIANT_NICE[v]} |"
add(header)
add("|---|" + "---|" * len(delta_variants))

wins = {v: 0 for v in delta_variants}
delta_means = {v: [] for v in delta_variants}

for site in SITE_ORDER:
    src = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == "source_448")]
    src_mean = src["dice_pos"].mean()
    deltas = {}
    for v in delta_variants:
        sub = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == v)]
        d = sub["dice_pos"].mean() - src_mean
        deltas[v] = d
        delta_means[v].append(d)
        if d > 0:
            wins[v] += 1
    best_v = max(deltas, key=lambda k: deltas[k])
    row = f"| {SITE_SHORT[site]} |"
    for v in delta_variants:
        d = deltas[v]
        cell = f"{d:+.4f}"
        if v == best_v:
            cell = f"**{cell}**"
        row += f" {cell} |"
    add(row)

row_mean = "| **Mean** |"
for v in delta_variants:
    row_mean += f" {np.mean(delta_means[v]):+.4f} |"
add(row_mean)

row_wins = "| **Wins** |"
for v in delta_variants:
    row_wins += f" {wins[v]}/6 |"
add(row_wins)
add("")

# ── Table 4: Delta vs uniform self-training ─────────────────────────────────
add("## Table 4: Delta vs Uniform Self-Training")
add("")
add("Shows marginal gain of agreement weighting over uniform pseudo-labels.")
add("")

agree_variants = ["ours_agree_224_448", "agree_448_448"]
header = "| Site | Uniform PL Dice | Agree 224-448 delta | Agree 448-448 delta |"
add(header)
add("|---|---|---|---|")

for site in SITE_ORDER:
    uni = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == "self_train_uniform")]
    uni_mean = uni["dice_pos"].mean()
    row = f"| {SITE_SHORT[site]} | {uni_mean:.4f} |"
    for v in agree_variants:
        sub = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == v)]
        d = sub["dice_pos"].mean() - uni_mean
        row += f" {d:+.4f} |"
    add(row)
add("")

# ── Table 5: FP rates and empty-mask counts ─────────────────────────────────
add("## Table 5: False-Positive Rates on Empty-Mask Images")
add("")
add("FP rate = fraction of empty-mask images where model predicts any foreground.")
add("n_neg = number of images with empty GT mask at that site.")
add("")

header = "| Site | n_neg |"
for v in VARIANT_ORDER:
    header += f" {VARIANT_NICE[v]} FP% |"
add(header)
add("|---|---|" + "---|" * len(VARIANT_ORDER))

for site in SITE_ORDER:
    sub0 = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == "source_448")]
    n_neg = int(sub0["n_neg"].iloc[0])
    row = f"| {SITE_SHORT[site]} | {n_neg} |"
    for v in VARIANT_ORDER:
        sub = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == v)]
        fp = sub["fp_rate"].mean() * 100
        row += f" {fp:.0f}% |"
    add(row)
add("")

# ── Table 6: All-images Dice ────────────────────────────────────────────────
add("## Table 6: All-Images Dice (mean over 5 seeds)")
add("")
add("Includes empty-mask images (Dice=1 if both pred and GT are empty, else 0).")
add("")

header = "| Site |"
for v in VARIANT_ORDER:
    header += f" {VARIANT_NICE[v]} |"
add(header)
add("|---|" + "---|" * len(VARIANT_ORDER))

for site in SITE_ORDER:
    row = f"| {SITE_SHORT[site]} |"
    for v in VARIANT_ORDER:
        sub = abl_df[(abl_df["site"] == site) & (abl_df["variant"] == v)]
        m = sub["dice_all"].mean()
        row += f" {m:.3f} |"
    add(row)
add("")

# ── Table 7: Cost ───────────────────────────────────────────────────────────
add("## Table 7: Adaptation Cost")
add("")
add("Adapt time = mean seconds per site for the self-training step (over 5 seeds).")
add("")

add("| Variant | Adapt time (s) | Extra params |")
add("|---|---|---|")
for v in VARIANT_ORDER:
    sub = abl_df[abl_df["variant"] == v]
    t = sub["adapt_time_s"].mean()
    params = "535,233" if t > 0 else "0"
    add(f"| {VARIANT_NICE[v]} | {t:.1f} | {params} |")
add("")

# ── Table 8: Few-label (224 features) ───────────────────────────────────────
add("## Table 8: Few-Label Supervised Fine-Tuning (224 features)")
add("")
add("Positives-only Dice, mean over 5 seeds. Decoder fine-tuned from Kvasir source weights.")
add("Features: DINOv2 ViT-S/14 at 224x224 (16x16 patch grid).")
add("")

few_df = pd.read_csv(ROOT / "outputs/08_controls/ctrl_fewlabel.csv")
k_vals = [0, 5, 10, 20, 50]
methods_k = {0: "source_only", 5: "finetune", 10: "finetune", 20: "finetune", 50: "finetune"}

add("| Site | Source-only | k=5 | k=10 | k=20 | k=50 |")
add("|---|---|---|---|---|---|")

for site in SITE_ORDER:
    row = f"| {SITE_SHORT[site]} |"
    for k in k_vals:
        method = methods_k[k]
        sub = few_df[(few_df["site"] == site) & (few_df["method"] == method) & (few_df["k"] == k)]
        if len(sub) == 0 and k == 0:
            sub = few_df[(few_df["site"] == site) & (few_df["method"] == "source_only")]
        if len(sub) > 0:
            m = sub["dice_pos"].mean()
            row += f" {m:.3f} |"
        else:
            row += " — |"
    add(row)
add("")

# ── PoC vs Ablation reconciliation note ─────────────────────────────────────
add("## Note: PoC vs Ablation Discrepancy")
add("")
add("The PoC script (poc_multires_5seeds.py) evaluates at decoder output resolution")
add("(128x128 for 224 features, 256x256 for 448 features). The ablation script")
add("(ablation_multires.py) evaluates at original GT mask resolution using bilinear")
add("upsampling of probabilities. The ablation numbers are the official results.")
add("Additionally, the PoC uses nearest-neighbor interpolation on binary masks for")
add("agreement computation, while the ablation uses bilinear interpolation on")
add("probabilities before thresholding.")
add("")

# ── Write output ────────────────────────────────────────────────────────────
out_path = OUT / "tables.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {out_path}")
print(f"{len(lines)} lines")

# Also print the tables to stdout for verification
print("\n" + "=" * 80)
print("\n".join(lines))
