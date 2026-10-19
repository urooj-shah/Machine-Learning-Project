"""Appearance shift check: pairwise domain classifiability between sites.

For every pair of sites, trains a logistic-regression domain classifier
on frozen DINOv2 features. Reports AUC matrix via 5-fold stratified CV.

Two feature variants: CLS token and mean-pooled patch tokens.

Caveat: AUC near 0.5 = visually indistinguishable, near 1.0 = easily
separable. Image-level separability can also come from borders/overlays,
not just scanner or population differences.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_features(feat_dir):
    """Load CLS and mean-pooled patch tokens per site."""
    sites = {}
    for npz_path in sorted(feat_dir.glob("*.npz")):
        name = npz_path.stem
        data = np.load(npz_path)
        cls = data["cls_tokens"].astype(np.float32)
        patches = data["patch_tokens"].astype(np.float32)
        mean_patch = patches.mean(axis=1)
        sites[name] = {"cls": cls, "mean_patch": mean_patch, "n": len(cls)}
    return sites


def domain_auc(feat_a, feat_b, n_splits=5):
    """Train logistic regression domain classifier, return mean AUC +/- std."""
    rng = np.random.default_rng(SEED)

    # Balance by subsampling the larger set
    n = min(len(feat_a), len(feat_b))
    idx_a = rng.choice(len(feat_a), size=n, replace=False)
    idx_b = rng.choice(len(feat_b), size=n, replace=False)

    X = np.concatenate([feat_a[idx_a], feat_b[idx_b]])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")
        clf.fit(X_train, y[train_idx])

        proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))

    return np.mean(aucs), np.std(aucs)


def build_matrix(sites, feat_key):
    """Build pairwise AUC matrix for a given feature type."""
    names = sorted(sites.keys())
    n = len(names)
    auc_mean = np.full((n, n), np.nan)
    auc_std = np.full((n, n), np.nan)

    for i, j in combinations(range(n), 2):
        m, s = domain_auc(sites[names[i]][feat_key], sites[names[j]][feat_key])
        auc_mean[i, j] = auc_mean[j, i] = m
        auc_std[i, j] = auc_std[j, i] = s

    np.fill_diagonal(auc_mean, 0.5)
    np.fill_diagonal(auc_std, 0.0)

    return names, auc_mean, auc_std


def save_heatmap(names, auc_mean, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(auc_mean, vmin=0.5, vmax=1.0, cmap="RdYlGn_r")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(len(names)):
        for j in range(len(names)):
            val = auc_mean[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Domain classifier AUC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", default=str(ROOT / "features" / "dinov2_vits14_224"))
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    out_dir = ROOT / "outputs" / "01_sanity_check" / "appearance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features ...")
    sites = load_features(feat_dir)

    print("\nSamples per site:")
    for name in sorted(sites):
        print(f"  {name}: n={sites[name]['n']}")

    for feat_key, label in [("cls", "CLS token"), ("mean_patch", "Mean-pooled patch tokens")]:
        print(f"\n{'='*60}")
        print(f"Feature: {label}")
        print(f"{'='*60}")

        names, auc_mean, auc_std = build_matrix(sites, feat_key)

        # Print matrix
        header = f"{'':>14}" + "".join(f"{n:>14}" for n in names)
        print(header)
        for i, name in enumerate(names):
            row = f"{name:>14}" + "".join(
                f"{auc_mean[i,j]:>10.3f}±{auc_std[i,j]:.3f}" if i != j
                else f"{'0.500':>14}"
                for j in range(len(names))
            )
            print(row)

        # Save CSV
        df = pd.DataFrame(auc_mean, index=names, columns=names).round(3)
        csv_path = out_dir / f"domain_auc_{feat_key}.csv"
        df.to_csv(csv_path)
        print(f"\nCSV: {csv_path}")

        # Save heatmap
        heatmap_path = out_dir / f"domain_auc_{feat_key}.png"
        save_heatmap(names, auc_mean, f"Domain classifier AUC — {label}", heatmap_path)
        print(f"Heatmap: {heatmap_path}")

    print("\nCaveat: AUC near 0.5 = visually indistinguishable; near 1.0 = easily")
    print("separable. Separability can also stem from borders/overlays, not just")
    print("scanner or population differences.")


if __name__ == "__main__":
    main()
