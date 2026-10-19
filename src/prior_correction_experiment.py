"""Test whether prior shift hurts a frozen-feature patch classifier,
and how much oracle / EM prior correction helps.

Train on Kvasir-SEG patch tokens, test on each PolypGen center.
Variants: (a) no correction, (b) oracle, (c) Saerens EM.
Run on all images and positive-only images.
Report Dice, IoU, calibration (ECE + reliability plots).
Few-label ablation: 10, 50, 200 Kvasir images x 5 seeds.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
OUT_DIR = ROOT / "outputs" / "03_main_experiments" / "oracle"
SPLIT_DIR = ROOT / "data" / "splits"
SEED = 42
PATCH_GRID = 16  # 224 / 14 = 16
N_PATCHES = PATCH_GRID * PATCH_GRID  # 256


# ── Data loading ─────────────────────────────────────────────────────

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


# ── Metrics ──────────────────────────────────────────────────────────

def dice_score(pred, gt):
    intersection = (pred & gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return 2 * intersection / (pred.sum() + gt.sum())


def iou_score(pred, gt):
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def compute_pixel_metrics(prob_patches, mask_path, orig_w, orig_h):
    """Upsample patch probabilities to original resolution, threshold, compute Dice/IoU."""
    prob_grid = prob_patches.reshape(PATCH_GRID, PATCH_GRID)
    prob_img = Image.fromarray(prob_grid, mode="F")
    prob_up = np.array(prob_img.resize((orig_w, orig_h), Image.BILINEAR))
    pred = (prob_up >= 0.5).astype(np.uint8)

    gt_mask = np.array(Image.open(mask_path).convert("L"))
    gt = (gt_mask >= 128).astype(np.uint8)

    return dice_score(pred, gt), iou_score(pred, gt)


# ── Calibration ──────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append((bins[i] + bins[i + 1]) / 2)
            bin_counts.append(0)
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
        bin_accs.append(acc)
        bin_confs.append(conf)
        bin_counts.append(mask.sum())
    ece /= len(probs)
    return ece, bin_accs, bin_confs, bin_counts


def save_reliability_plot(site_name, probs, labels, ece, out_path):
    _, bin_accs, bin_confs, bin_counts = expected_calibration_error(probs, labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    valid = [i for i, c in enumerate(bin_counts) if c > 0]
    ax.bar([bin_confs[i] for i in valid], [bin_accs[i] for i in valid],
           width=0.06, alpha=0.6, edgecolor="black", label="Observed")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{site_name} — ECE={ece:.4f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close()


# ── Prior correction ─────────────────────────────────────────────────

def adjust_logits(logits, pi_source, pi_target):
    """Shift logits by log-odds difference to correct for prior shift.
    logit_corrected = logit + log(pi_t / (1-pi_t)) - log(pi_s / (1-pi_s))
    """
    eps = 1e-7
    pi_s = np.clip(pi_source, eps, 1 - eps)
    pi_t = np.clip(pi_target, eps, 1 - eps)
    shift = np.log(pi_t / (1 - pi_t)) - np.log(pi_s / (1 - pi_s))
    return logits + shift


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def saerens_em(probs, max_iter=100, tol=1e-6):
    """Saerens et al. EM for estimating target prior from unlabeled probs.
    Returns corrected probabilities and estimated target prior.
    """
    pi_s = probs.mean()
    pi_t = pi_s
    p = probs.copy()

    for _ in range(max_iter):
        # E-step: adjust posteriors
        ratio_pos = pi_t / (pi_s + 1e-10)
        ratio_neg = (1 - pi_t) / (1 - pi_s + 1e-10)
        numerator = p * ratio_pos
        denominator = numerator + (1 - p) * ratio_neg
        q = numerator / (denominator + 1e-10)

        # M-step: update target prior
        pi_t_new = q.mean()
        if abs(pi_t_new - pi_t) < tol:
            pi_t = pi_t_new
            break
        pi_t = pi_t_new

    return q, pi_t


# ── Train and evaluate ───────────────────────────────────────────────

def train_source_classifier(kvasir_data, train_indices):
    """Train logistic regression on Kvasir patch tokens."""
    X_all = kvasir_data["patch_tokens"][train_indices].reshape(-1, kvasir_data["patch_tokens"].shape[-1])
    y_all = kvasir_data["patch_fg_binary"][train_indices].reshape(-1)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs",
                             class_weight="balanced")
    clf.fit(X_all, y_all)

    # Source patch-level prior (from labels, NOT from class_weight-adjusted model)
    pi_source = y_all.mean()

    return clf, pi_source


def evaluate_site(clf, pi_source, site_data, site_name, positive_only, label_tag=""):
    """Evaluate on a target site with no correction, oracle, and EM."""
    X = site_data["patch_tokens"].reshape(-1, site_data["patch_tokens"].shape[-1])
    y = site_data["patch_fg_binary"].reshape(-1)

    # Raw logits and probabilities from the classifier
    # class_weight="balanced" changes the decision function, so the raw
    # predict_proba already incorporates reweighting. The logits from
    # decision_function are the balanced logits. To get calibrated probs
    # for Saerens EM we use predict_proba (Platt-scaled by sklearn).
    logits_raw = clf.decision_function(X)
    probs_raw = clf.predict_proba(X)[:, 1]

    # For the oracle/EM correction we work in logit space.
    # sklearn's balanced logistic regression shifts the intercept by
    # log(w1/w0) where w1, w0 are the class weights. The predict_proba
    # already accounts for this. We correct the raw decision_function
    # logits. The "source prior" for logit correction should be the
    # effective prior after balancing = 0.5 (since balanced weights make
    # the model act as if classes are 50/50).
    pi_source_effective = 0.5

    # Target true prior (patch level)
    indices = np.arange(len(site_data["filenames"]))
    if positive_only:
        pos_mask = site_data["patch_fg_binary"].sum(axis=1) > 0
        indices = np.where(pos_mask)[0]

    results = []

    for method in ["no_correction", "oracle", "saerens_em"]:
        dices = []
        ious = []

        if method == "oracle":
            # True target prior from labels
            target_labels = site_data["patch_fg_binary"][indices].reshape(-1)
            pi_target = target_labels.mean()
            logits_adj = adjust_logits(logits_raw, pi_source_effective, pi_target)
            probs_eval = sigmoid(logits_adj)
        elif method == "saerens_em":
            # EM on target unlabeled patches
            target_probs_raw = probs_raw.copy()
            _, pi_target_em = saerens_em(target_probs_raw)
            logits_adj = adjust_logits(logits_raw, pi_source_effective, pi_target_em)
            probs_eval = sigmoid(logits_adj)
        else:
            probs_eval = probs_raw
            pi_target_em = None

        for idx in indices:
            p = probs_eval[idx * N_PATCHES:(idx + 1) * N_PATCHES]
            mask_path = str(site_data["mask_paths"][idx])
            orig_w, orig_h = site_data["orig_sizes"][idx]
            d, iou = compute_pixel_metrics(p, mask_path, int(orig_w), int(orig_h))
            dices.append(d)
            ious.append(iou)

        row = {
            "site": site_name,
            "method": method,
            "subset": "positive_only" if positive_only else "all",
            "n_images": len(indices),
            "dice_mean": np.mean(dices),
            "dice_std": np.std(dices),
            "iou_mean": np.mean(ious),
            "iou_std": np.std(ious),
        }

        if method == "oracle":
            row["pi_target_true"] = float(site_data["patch_fg_binary"][indices].reshape(-1).mean())
        elif method == "saerens_em":
            row["pi_target_true"] = float(site_data["patch_fg_binary"][indices].reshape(-1).mean())
            row["pi_target_em"] = float(pi_target_em)

        if label_tag:
            row["label_tag"] = label_tag

        results.append(row)

    return results, probs_raw, y


def run_calibration(clf, site_data, site_name, indices, out_dir):
    """Compute ECE and save reliability plot."""
    X = site_data["patch_tokens"][indices].reshape(-1, site_data["patch_tokens"].shape[-1])
    y = site_data["patch_fg_binary"][indices].reshape(-1)
    probs = clf.predict_proba(X)[:, 1]
    ece, _, _, _ = expected_calibration_error(probs, y)

    cal_dir = out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    save_reliability_plot(site_name, probs, y, ece, cal_dir / f"reliability_{site_name}.png")

    return ece


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Kvasir ...")
    kvasir = load_site("kvasir")

    # Train/val split (80/20 by image)
    n_kvasir = len(kvasir["filenames"])
    all_idx = np.arange(n_kvasir)
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=SEED)

    split_info = {
        "train": kvasir["filenames"][train_idx].tolist(),
        "val": kvasir["filenames"][val_idx].tolist(),
    }
    with open(SPLIT_DIR / "kvasir_train_val.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Train classifier
    print("Training patch classifier on Kvasir train split ...")
    clf, pi_source = train_source_classifier(kvasir, train_idx)
    print(f"  Source patch-level fg rate: {pi_source:.4f}")
    print(f"  Classifier uses class_weight='balanced', effective prior = 0.5")

    # Load all target sites
    target_names = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                    "polypgen_C4", "polypgen_C5", "polypgen_C6"]
    targets = {}
    for name in target_names:
        targets[name] = load_site(name)

    # ── Calibration ──────────────────────────────────────────────────
    print("\n=== Calibration ===")
    cal_rows = []

    # Kvasir validation
    ece_kv = run_calibration(clf, kvasir, "kvasir_val", val_idx, OUT_DIR)
    cal_rows.append({"site": "kvasir_val", "ECE": ece_kv})
    print(f"  kvasir_val ECE: {ece_kv:.4f}")

    for name in target_names:
        all_idx_t = np.arange(len(targets[name]["filenames"]))
        ece = run_calibration(clf, targets[name], name, all_idx_t, OUT_DIR)
        cal_rows.append({"site": name, "ECE": ece})
        print(f"  {name} ECE: {ece:.4f}")

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT_DIR / "calibration_ece.csv", index=False)

    if any(r["ECE"] > 0.1 for r in cal_rows):
        print("  WARNING: ECE > 0.1 on some sites. Saerens EM assumes calibrated")
        print("  probabilities — results may be unreliable where ECE is large.")

    # ── Main experiment (a, b, c) ────────────────────────────────────
    print("\n=== Main experiment ===")
    all_results = []

    for positive_only in [False, True]:
        tag = "positive_only" if positive_only else "all"
        print(f"\n--- Subset: {tag} ---")

        # Kvasir validation baseline
        kv_results, _, _ = evaluate_site(clf, pi_source, kvasir, "kvasir_val", positive_only)
        # Only keep val images
        kv_val_data = {k: kvasir[k][val_idx] if hasattr(kvasir[k], '__getitem__') and kvasir[k].ndim > 0 else kvasir[k]
                       for k in kvasir}
        kv_val_data["patch_tokens"] = kvasir["patch_tokens"][val_idx]
        kv_val_data["patch_fg_binary"] = kvasir["patch_fg_binary"][val_idx]
        kv_val_data["patch_fg_fraction"] = kvasir["patch_fg_fraction"][val_idx]
        kv_val_data["filenames"] = kvasir["filenames"][val_idx]
        kv_val_data["mask_paths"] = kvasir["mask_paths"][val_idx]
        kv_val_data["orig_sizes"] = kvasir["orig_sizes"][val_idx]

        kv_val_results, _, _ = evaluate_site(clf, pi_source, kv_val_data, "kvasir_val", positive_only)
        all_results.extend(kv_val_results)

        for name in target_names:
            site_results, _, _ = evaluate_site(clf, pi_source, targets[name], name, positive_only)
            all_results.extend(site_results)

    # Print results table
    df = pd.DataFrame(all_results)
    for subset in ["all", "positive_only"]:
        print(f"\n{'='*80}")
        print(f"Subset: {subset}")
        print(f"{'='*80}")
        sub = df[df["subset"] == subset].copy()
        pivot_dice = sub.pivot(index="site", columns="method", values="dice_mean")
        pivot_dice = pivot_dice[["no_correction", "oracle", "saerens_em"]]
        pivot_dice.columns = ["Dice(none)", "Dice(oracle)", "Dice(EM)"]

        # Add pi columns
        em_rows = sub[sub["method"] == "saerens_em"][["site", "pi_target_true", "pi_target_em"]].set_index("site")
        pivot_dice = pivot_dice.join(em_rows)

        n_col = sub[sub["method"] == "no_correction"][["site", "n_images"]].set_index("site")
        pivot_dice = n_col.join(pivot_dice)

        print(pivot_dice.round(4).to_string())

    df.to_csv(OUT_DIR / "main_results.csv", index=False)
    print(f"\nResults saved to {OUT_DIR / 'main_results.csv'}")

    # ── Few-label ablation ───────────────────────────────────────────
    print("\n=== Few-label ablation ===")
    label_counts = [10, 50, 200]
    n_seeds = 5
    ablation_results = []

    for n_labels in label_counts:
        for seed in range(n_seeds):
            rng = np.random.default_rng(SEED + seed)
            sub_idx = rng.choice(train_idx, size=min(n_labels, len(train_idx)), replace=False)

            clf_sub, pi_sub = train_source_classifier(kvasir, sub_idx)

            for name in target_names:
                for positive_only in [False, True]:
                    subset_tag = "positive_only" if positive_only else "all"
                    site_results, _, _ = evaluate_site(
                        clf_sub, pi_sub, targets[name], name, positive_only
                    )
                    for r in site_results:
                        r["n_labels"] = n_labels
                        r["seed"] = seed
                        ablation_results.append(r)

        print(f"  n_labels={n_labels} done")

    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv(OUT_DIR / "few_label_ablation.csv", index=False)

    # Also add full-training results for the plot
    full_results = []
    for name in target_names:
        for positive_only in [False, True]:
            site_results, _, _ = evaluate_site(
                clf, pi_source, targets[name], name, positive_only
            )
            for r in site_results:
                r["n_labels"] = len(train_idx)
                r["seed"] = 0
                full_results.append(r)

    full_df = pd.DataFrame(full_results)
    combined_abl = pd.concat([abl_df, full_df], ignore_index=True)

    # Plot: Dice vs n_labels for no_correction and saerens_em (all images)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax_idx, name in enumerate(target_names):
        ax = axes[ax_idx // 3, ax_idx % 3]
        for method, color, marker in [("no_correction", "#c44e52", "o"), ("saerens_em", "#4c72b0", "s")]:
            sub = combined_abl[(combined_abl["site"] == name) &
                               (combined_abl["method"] == method) &
                               (combined_abl["subset"] == "all")]
            grouped = sub.groupby("n_labels")["dice_mean"].agg(["mean", "std"]).reset_index()
            ax.errorbar(grouped["n_labels"], grouped["mean"], yerr=grouped["std"],
                        marker=marker, label=method, color=color, capsize=3, markersize=5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("# Kvasir labels")
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 200, len(train_idx)])
        ax.set_xticklabels(["10", "50", "200", "800"])
        ax.grid(alpha=0.3)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Dice")
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Dice vs # Kvasir training labels (all images)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "few_label_dice.png", dpi=150)
    plt.close()
    print(f"Few-label plot: {OUT_DIR / 'few_label_dice.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
