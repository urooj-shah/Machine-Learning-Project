"""Prior correction experiment using the trained patch decoder.

Train on Kvasir, test on each PolypGen center.
Variants: (a) no correction, (b) oracle, (c) Saerens EM.
Runs on all images and positive-only. Reports Dice, IoU, calibration.
Few-label ablation: 10, 50, 200 labels x 5 seeds.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from patch_decoder import PatchDecoder

ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs" / "03_main_experiments" / "decoder"

GRID = 16
MASK_SIZE = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


# ── Data ─────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(384, GRID, GRID)
        y = self.masks[idx]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


def load_site(name):
    d = np.load(FEAT_DIR / f"{name}.npz")
    patch_tokens = d["patch_tokens"].astype(np.float32)
    mask_paths = d["mask_paths"]
    orig_sizes = d["orig_sizes"]
    fg_binary = d["patch_fg_binary"]

    masks_128 = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
        masks_128.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    masks_128 = np.stack(masks_128)

    return {
        "patch_tokens": patch_tokens,
        "masks_128": masks_128,
        "mask_paths": mask_paths,
        "orig_sizes": orig_sizes,
        "patch_fg_binary": fg_binary,
    }


# ── Metrics ──────────────────────────────────────────────────────────

def pixel_dice_iou(prob_128, mask_path, orig_w, orig_h):
    prob_img = Image.fromarray(prob_128, mode="F")
    prob_up = np.array(prob_img.resize((orig_w, orig_h), Image.BILINEAR))
    pred = (prob_up >= 0.5).astype(np.uint8)

    gt = (np.array(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)

    inter = (pred & gt).sum()
    union_d = pred.sum() + gt.sum()
    union_i = (pred | gt).sum()

    dice = 2 * inter / union_d if union_d > 0 else 1.0
    iou = inter / union_i if union_i > 0 else 1.0
    return dice, iou


# ── Calibration ──────────────────────────────────────────────────────

def ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        n = mask.sum()
        if n == 0:
            bin_data.append((np.nan, (bins[i] + bins[i + 1]) / 2, 0))
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        total += n * abs(acc - conf)
        bin_data.append((acc, conf, n))
    return total / len(probs), bin_data


def save_reliability(name, probs, labels, ece_val, path):
    _, bin_data = ece(probs, labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    valid = [(acc, conf) for acc, conf, n in bin_data if n > 0]
    if valid:
        ax.bar([c for _, c in valid], [a for a, _ in valid],
               width=0.06, alpha=0.6, edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(f"{name} — ECE={ece_val:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close()


# ── Prior correction ─────────────────────────────────────────────────

def logit_shift(pi_source, pi_target):
    eps = 1e-7
    ps = np.clip(pi_source, eps, 1 - eps)
    pt = np.clip(pi_target, eps, 1 - eps)
    return np.log(pt / (1 - pt)) - np.log(ps / (1 - ps))


def saerens_em(probs, max_iter=100, tol=1e-6):
    pi_s = probs.mean()
    pi_t = pi_s
    p = probs.copy()
    for _ in range(max_iter):
        r_pos = pi_t / (pi_s + 1e-10)
        r_neg = (1 - pi_t) / (1 - pi_s + 1e-10)
        num = p * r_pos
        q = num / (num + (1 - p) * r_neg + 1e-10)
        pi_new = q.mean()
        if abs(pi_new - pi_t) < tol:
            break
        pi_t = pi_new
    return pi_t


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Train decoder ────────────────────────────────────────────────────

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder(train_tokens, train_masks, n_epochs=80, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    ds = FeatureDataset(train_tokens, train_masks)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    model = PatchDecoder(in_dim=384).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(n_epochs):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = bce_fn(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    pi_source = train_masks.mean()
    return model, pi_source


# ── Evaluate site ────────────────────────────────────────────────────

@torch.no_grad()
def get_logits_128(model, patch_tokens):
    """Get 128x128 logit maps for all images."""
    model.eval()
    all_logits = []
    ds = torch.from_numpy(patch_tokens.reshape(-1, 384, GRID, GRID))
    for i in range(0, len(ds), 32):
        batch = ds[i:i+32].to(DEVICE)
        out = model(batch).cpu().numpy()[:, 0]
        all_logits.append(out)
    return np.concatenate(all_logits)


def evaluate_site(model, pi_source, site, site_name, positive_only):
    logits_128 = get_logits_128(model, site["patch_tokens"])
    probs_128 = sigmoid(logits_128)

    indices = np.arange(len(site["mask_paths"]))
    if positive_only:
        pos = site["patch_fg_binary"].sum(axis=1) > 0
        indices = np.where(pos)[0]

    # True target prior at 128x128
    pi_target_true = site["masks_128"][indices].mean()

    # Saerens EM on ALL target images (unlabeled)
    all_probs_flat = probs_128.reshape(-1)
    pi_target_em = saerens_em(all_probs_flat)

    shift_oracle = logit_shift(pi_source, pi_target_true)
    shift_em = logit_shift(pi_source, pi_target_em)

    results = []
    for method in ["no_correction", "oracle", "saerens_em"]:
        dices, ious = [], []
        for idx in indices:
            if method == "oracle":
                p = sigmoid(logits_128[idx] + shift_oracle)
            elif method == "saerens_em":
                p = sigmoid(logits_128[idx] + shift_em)
            else:
                p = probs_128[idx]

            orig_w, orig_h = site["orig_sizes"][idx]
            d, iou = pixel_dice_iou(p, str(site["mask_paths"][idx]), int(orig_w), int(orig_h))
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
            "pi_source": float(pi_source),
            "pi_target_true": float(pi_target_true),
            "pi_target_em": float(pi_target_em),
        }
        results.append(row)

    return results, probs_128, logits_128


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration").mkdir(exist_ok=True)

    # Load Kvasir and split
    print("Loading Kvasir ...")
    kvasir = load_site("kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    filenames = list(np.load(FEAT_DIR / "kvasir.npz")["filenames"])
    train_idx = np.array([filenames.index(fn) for fn in split["train"]])
    val_idx = np.array([filenames.index(fn) for fn in split["val"]])

    # Load pretrained decoder
    print("Loading pretrained decoder ...")
    ckpt = torch.load(CKPT_DIR / "decoder_best.pt", map_location=DEVICE, weights_only=False)
    model = PatchDecoder(in_dim=384).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    pi_source = ckpt["pi_source"]
    print(f"Source prior: {pi_source:.4f}, val Dice from training: {ckpt['val_dice']:.4f}")

    # Load target sites
    print("Loading target sites ...")
    targets = {}
    for name in TARGET_SITES:
        targets[name] = load_site(name)
        print(f"  {name}: {len(targets[name]['mask_paths'])} images")

    # Build kvasir_val site dict
    kv_val = {k: kvasir[k][val_idx] for k in kvasir}

    # ── Calibration ──────────────────────────────────────────────────
    print("\n=== Calibration ===")
    cal_rows = []
    for name, site_data, idx_set in [("kvasir_val", kvasir, val_idx)] + [(n, targets[n], None) for n in TARGET_SITES]:
        if idx_set is not None:
            tokens = site_data["patch_tokens"][idx_set]
            masks = site_data["masks_128"][idx_set]
        else:
            tokens = site_data["patch_tokens"]
            masks = site_data["masks_128"]

        logits = get_logits_128(model, tokens)
        probs_flat = sigmoid(logits).reshape(-1)
        labels_flat = masks.reshape(-1)

        ece_val, _ = ece(probs_flat, labels_flat)
        cal_rows.append({"site": name, "ECE": ece_val})
        print(f"  {name}: ECE={ece_val:.4f}")
        save_reliability(name, probs_flat, labels_flat, ece_val,
                         OUT_DIR / "calibration" / f"reliability_{name}.png")

    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "calibration_ece.csv", index=False)
    if any(r["ECE"] > 0.10 for r in cal_rows):
        print("  WARNING: ECE > 0.10 detected. Saerens EM may be unreliable.")

    # ── Main experiment ──────────────────────────────────────────────
    print("\n=== Main experiment ===")
    all_results = []

    for positive_only in [False, True]:
        tag = "positive_only" if positive_only else "all"

        # Kvasir val
        res, _, _ = evaluate_site(model, pi_source, kv_val, "kvasir_val", positive_only)
        all_results.extend(res)

        for name in TARGET_SITES:
            res, _, _ = evaluate_site(model, pi_source, targets[name], name, positive_only)
            all_results.extend(res)

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / "main_results.csv", index=False)

    for subset in ["all", "positive_only"]:
        print(f"\n{'='*90}")
        print(f"Subset: {subset}")
        print(f"{'='*90}")
        sub = df[df["subset"] == subset]
        pivot = sub.pivot(index="site", columns="method", values="dice_mean")
        pivot = pivot[["no_correction", "oracle", "saerens_em"]]
        pivot.columns = ["Dice(none)", "Dice(oracle)", "Dice(EM)"]

        iou_p = sub.pivot(index="site", columns="method", values="iou_mean")
        pivot["IoU(none)"] = iou_p["no_correction"]

        em = sub[sub["method"] == "saerens_em"][["site", "pi_source", "pi_target_true", "pi_target_em"]].set_index("site")
        pivot = pivot.join(em)

        n_col = sub[sub["method"] == "no_correction"][["site", "n_images"]].set_index("site")
        pivot = n_col.join(pivot)

        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.width", 130)
        print(pivot.to_string())

    # ── Few-label ablation ───────────────────────────────────────────
    print("\n=== Few-label ablation ===")
    label_counts = [10, 50, 200]
    n_seeds = 5
    abl_results = []

    for n_labels in label_counts:
        for seed in range(n_seeds):
            rng = np.random.default_rng(SEED + seed)
            sub_idx = rng.choice(train_idx, size=min(n_labels, len(train_idx)), replace=False)

            sub_model, sub_pi = train_decoder(
                kvasir["patch_tokens"][sub_idx],
                kvasir["masks_128"][sub_idx],
                n_epochs=80, seed=SEED + seed,
            )

            for name in TARGET_SITES:
                for pos_only in [False, True]:
                    res, _, _ = evaluate_site(sub_model, sub_pi, targets[name], name, pos_only)
                    for r in res:
                        r["n_labels"] = n_labels
                        r["seed"] = seed
                    abl_results.extend(res)

        print(f"  n_labels={n_labels} done")

    # Add full-training results
    for name in TARGET_SITES:
        for pos_only in [False, True]:
            res, _, _ = evaluate_site(model, pi_source, targets[name], name, pos_only)
            for r in res:
                r["n_labels"] = len(train_idx)
                r["seed"] = 0
            abl_results.extend(res)

    abl_df = pd.DataFrame(abl_results)
    abl_df.to_csv(OUT_DIR / "few_label_ablation.csv", index=False)

    # ── Plot: Dice vs n_labels ───────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax_idx, name in enumerate(TARGET_SITES):
        ax = axes[ax_idx // 3, ax_idx % 3]
        for method, color, marker, label in [
            ("no_correction", "#c44e52", "o", "No correction"),
            ("saerens_em", "#4c72b0", "s", "Saerens EM"),
        ]:
            sub = abl_df[(abl_df["site"] == name) &
                         (abl_df["method"] == method) &
                         (abl_df["subset"] == "all")]
            grp = sub.groupby("n_labels")["dice_mean"].agg(["mean", "std"]).reset_index()
            ax.errorbar(grp["n_labels"], grp["mean"], yerr=grp["std"],
                        marker=marker, label=label, color=color, capsize=3, markersize=5)
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

    fig.suptitle("Dice vs # Kvasir labels — decoder (all images)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "few_label_dice.png", dpi=150)
    plt.close()
    print(f"\nPlot: {OUT_DIR / 'few_label_dice.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
