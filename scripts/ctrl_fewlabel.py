"""Control 6: Few-label target adaptation.

For each PolypGen center, sample k in {1,5,10,20,50} labeled target images
(5 seeds), fine-tune ONLY the decoder initialized from Kvasir weights for
a fixed number of steps chosen on Kvasir val; evaluate on remaining images.

Compare with:
  (a) source-only decoder
  (b) decoder trained from scratch on k images

Add pseudo-label self-training on unlabeled remainder for k in {0,5,10}
(confidence threshold fixed a priori, one round).

Uses 224x224 features (or 448 if available and better — checked at runtime).
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import GRID, N_PATCHES, bootstrap_ci, dice_score, load_site_features, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
FEAT_DIR_448 = ROOT / "features" / "dinov2_vits14_448"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FINETUNE_STEPS = 200  # fixed a priori, tuned below on Kvasir val
FINETUNE_LR = 5e-4
K_VALUES = [1, 5, 10, 20, 50]
SEEDS = [42, 43, 44, 45, 46]
PSEUDO_K = [0, 5, 10]
PSEUDO_CONF = 0.9  # confidence threshold for pseudo-labels

TARGET_SITES = [f"polypgen_C{i}" for i in range(1, 7)]

# Check if 448 features exist and are better
USE_448 = False
GRID_USED = GRID
FEAT_DIR_USED = FEAT_DIR


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks, grid=16):
        self.tokens = patch_tokens
        self.masks = masks
        self.grid = grid
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(self.grid, self.grid, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


def load_masks(mask_paths, size=128):
    masks = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((size, size), Image.NEAREST)
        masks.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(masks)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_source_decoder(train_tokens, train_masks, val_tokens, val_masks,
                         n_epochs=80, lr=1e-3, seed=42, grid=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatureDataset(train_tokens, train_masks, grid)
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
            loss = bce(model(x), y) + dice_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        vds = FeatureDataset(val_tokens, val_masks, grid)
        vdl = DataLoader(vds, batch_size=32, shuffle=False, num_workers=0)
        td = 0
        with torch.no_grad():
            for x, y in vdl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                inter = (p * y).sum(dim=(1, 2, 3))
                union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                td += ((2 * inter + 1) / (union + 1)).sum().item()
        vd = td / len(vds)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best_dice


def finetune_decoder(base_model, train_tokens, train_masks, n_steps, lr, grid=16):
    model = copy.deepcopy(base_model)
    ds = FeatureDataset(train_tokens, train_masks, grid)
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    while step < n_steps:
        for x, y in dl:
            if step >= n_steps:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = bce(model(x), y) + dice_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    model.eval()
    return model


def train_scratch(train_tokens, train_masks, n_steps, lr, grid=16):
    model = PatchDecoder(in_dim=384).to(DEVICE)
    ds = FeatureDataset(train_tokens, train_masks, grid)
    dl = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    while step < n_steps:
        for x, y in dl:
            if step >= n_steps:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = bce(model(x), y) + dice_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
    model.eval()
    return model


@torch.no_grad()
def decoder_logits(model, tokens, grid=16):
    model.eval()
    out = []
    for i in range(0, len(tokens), 64):
        batch = tokens[i:i + 64]
        x = np.stack([t.reshape(grid, grid, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_dice_128(model, tokens, gt_128, grid=16):
    logits = decoder_logits(model, tokens, grid)
    probs = sigmoid(logits)
    dices = []
    for i in range(len(tokens)):
        pred = (probs[i] >= 0.5).astype(np.uint8)
        gt = (gt_128[i] >= 0.5).astype(np.uint8)
        is_empty = gt.sum() == 0
        if not is_empty:
            dices.append(dice_score(pred, gt))
    return np.array(dices)


@torch.no_grad()
def generate_pseudo_labels(model, tokens, conf_thresh, grid=16, mask_size=128):
    logits = decoder_logits(model, tokens, grid)
    probs = sigmoid(logits)
    pseudo_masks = []
    confident = []
    for i in range(len(tokens)):
        p = probs[i]
        high_conf = (p >= conf_thresh) | (p <= (1 - conf_thresh))
        frac_conf = high_conf.mean()
        mask = (p >= 0.5).astype(np.float32)
        pseudo_masks.append(mask)
        confident.append(frac_conf)
    return np.stack(pseudo_masks), np.array(confident)


def tune_finetune_steps(source_model, kv_train_tokens, kv_train_masks,
                        kv_val_tokens, kv_val_masks, grid=16):
    """Pick finetune steps: hold out 50 images from kvasir train as 'target',
    finetune with k=10 from them, eval on the rest."""
    rng = np.random.default_rng(42)
    n = len(kv_train_tokens)
    perm = rng.permutation(n)
    pseudo_target = perm[:50]
    pseudo_rest = perm[50:]
    pt_tokens = kv_train_tokens[pseudo_target[:10]]
    pt_masks = kv_train_masks[pseudo_target[:10]]
    eval_tokens = kv_train_tokens[pseudo_target[10:]]
    eval_masks = kv_train_masks[pseudo_target[10:]]

    best_steps, best_dice = 200, 0
    for steps in [50, 100, 200, 400]:
        model = finetune_decoder(source_model, pt_tokens, pt_masks, steps, FINETUNE_LR, grid)
        d = eval_dice_128(model, eval_tokens, eval_masks, grid)
        md = d.mean() if len(d) > 0 else 0
        if md > best_dice:
            best_dice = md
            best_steps = steps
    return best_steps


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    grid = GRID
    mask_size = MASK_SIZE

    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks(kvasir["mask_paths"], mask_size)

    print("Training source decoder ...", flush=True)
    ts = time.time()
    source_model, vd = train_source_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=42, grid=grid)
    print(f"  val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)

    print("Tuning finetune steps on Kvasir val ...", flush=True)
    ft_steps = tune_finetune_steps(
        source_model, kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx], grid)
    print(f"  Selected finetune steps: {ft_steps}", flush=True)

    # Load targets
    targets = {nm: load_site_features(FEAT_DIR, nm) for nm in TARGET_SITES}
    target_m = {nm: load_masks(targets[nm]["mask_paths"], mask_size) for nm in TARGET_SITES}

    rows = []
    for site_name in TARGET_SITES:
        sd = targets[site_name]
        n_total = len(sd["filenames"])
        gt_all = target_m[site_name]
        tokens_all = sd["patch_tokens"]
        print(f"\n  {site_name} ({n_total}) ...", flush=True)

        # Source-only baseline
        d_src = eval_dice_128(source_model, tokens_all, gt_all, grid)
        rows.append({
            "site": site_name, "method": "source_only", "k": 0, "seed": -1,
            "dice_pos": float(d_src.mean()) if len(d_src) > 0 else float("nan"),
            "n_eval": n_total, "train_s": 0,
        })
        print(f"    source_only: {d_src.mean():.4f}", flush=True)

        for k in K_VALUES:
            if k >= n_total:
                continue
            dice_ft_seeds = []
            dice_scratch_seeds = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n_total)
                train_k = perm[:k]
                test_k = perm[k:]

                tok_train = tokens_all[train_k]
                gt_train = gt_all[train_k]
                tok_test = tokens_all[test_k]
                gt_test = gt_all[test_k]

                # Fine-tune from source
                t_ft = time.time()
                model_ft = finetune_decoder(source_model, tok_train, gt_train, ft_steps, FINETUNE_LR, grid)
                t_ft = time.time() - t_ft
                d_ft = eval_dice_128(model_ft, tok_test, gt_test, grid)
                dice_ft = float(d_ft.mean()) if len(d_ft) > 0 else float("nan")
                dice_ft_seeds.append(dice_ft)
                rows.append({
                    "site": site_name, "method": "finetune", "k": k, "seed": seed,
                    "dice_pos": dice_ft, "n_eval": len(test_k), "train_s": round(t_ft, 2),
                })

                # Train from scratch
                t_sc = time.time()
                model_scratch = train_scratch(tok_train, gt_train, ft_steps, FINETUNE_LR, grid)
                t_sc = time.time() - t_sc
                d_sc = eval_dice_128(model_scratch, tok_test, gt_test, grid)
                dice_sc = float(d_sc.mean()) if len(d_sc) > 0 else float("nan")
                dice_scratch_seeds.append(dice_sc)
                rows.append({
                    "site": site_name, "method": "scratch", "k": k, "seed": seed,
                    "dice_pos": dice_sc, "n_eval": len(test_k), "train_s": round(t_sc, 2),
                })

            print(f"    k={k}: finetune={np.nanmean(dice_ft_seeds):.4f}±{np.nanstd(dice_ft_seeds):.4f} "
                  f"scratch={np.nanmean(dice_scratch_seeds):.4f}±{np.nanstd(dice_scratch_seeds):.4f}", flush=True)

    # Pseudo-label self-training
    print("\n=== Pseudo-label self-training ===", flush=True)
    pseudo_rows = []
    for site_name in TARGET_SITES:
        sd = targets[site_name]
        n_total = len(sd["filenames"])
        gt_all = target_m[site_name]
        tokens_all = sd["patch_tokens"]
        print(f"\n  {site_name} ...", flush=True)

        for k in PSEUDO_K:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n_total)

                if k == 0:
                    # Pure pseudo-label: use source model
                    labeled_tokens = np.empty((0, N_PATCHES, 384), dtype=np.float32)
                    labeled_masks = np.empty((0, mask_size, mask_size), dtype=np.float32)
                    unlabeled_idx = perm
                    base_model = source_model
                else:
                    train_k = perm[:k]
                    unlabeled_idx = perm[k:]
                    labeled_tokens = tokens_all[train_k]
                    labeled_masks = gt_all[train_k]
                    base_model = finetune_decoder(source_model, labeled_tokens, labeled_masks,
                                                  ft_steps, FINETUNE_LR, grid)

                # Generate pseudo-labels for unlabeled set
                unlabeled_tokens = tokens_all[unlabeled_idx]
                pseudo_masks, conf_fracs = generate_pseudo_labels(
                    base_model, unlabeled_tokens, PSEUDO_CONF, grid, mask_size)

                # Combine labeled + pseudo-labeled
                if k > 0:
                    combined_tokens = np.concatenate([labeled_tokens, unlabeled_tokens])
                    combined_masks = np.concatenate([labeled_masks, pseudo_masks])
                else:
                    combined_tokens = unlabeled_tokens
                    combined_masks = pseudo_masks

                # Fine-tune
                t_ps = time.time()
                model_ps = finetune_decoder(source_model if k == 0 else base_model,
                                            combined_tokens, combined_masks,
                                            ft_steps, FINETUNE_LR, grid)
                t_ps = time.time() - t_ps

                # Eval on full set (since pseudo-labels are from unlabeled)
                d_ps = eval_dice_128(model_ps, tokens_all, gt_all, grid)
                dice_ps = float(d_ps.mean()) if len(d_ps) > 0 else float("nan")

                pseudo_rows.append({
                    "site": site_name, "method": f"pseudo_k{k}", "k": k, "seed": seed,
                    "dice_pos": dice_ps, "n_pseudo": len(unlabeled_idx),
                    "mean_conf": float(conf_fracs.mean()), "train_s": round(t_ps, 2),
                })

            agg = [r["dice_pos"] for r in pseudo_rows if r["site"] == site_name and r["k"] == k]
            print(f"    pseudo k={k}: {np.nanmean(agg):.4f}±{np.nanstd(agg):.4f}", flush=True)

    # Output
    df = pd.DataFrame(rows)
    df_pseudo = pd.DataFrame(pseudo_rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    # Summary table
    print("\n" + "=" * 120)
    print(f"TABLE: FEW-LABEL ADAPTATION (decoder, {ft_steps} steps, 128x128 eval)")
    print("=" * 120)
    summary = df.groupby(["site", "method", "k"]).agg(
        dice_mean=("dice_pos", "mean"),
        dice_std=("dice_pos", "std"),
        n_seeds=("dice_pos", "count"),
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE: PSEUDO-LABEL SELF-TRAINING")
    print("=" * 100)
    ps_summary = df_pseudo.groupby(["site", "method", "k"]).agg(
        dice_mean=("dice_pos", "mean"),
        dice_std=("dice_pos", "std"),
    ).reset_index()
    print(ps_summary.to_string(index=False))

    df.to_csv(OUT_DIR / "ctrl_fewlabel.csv", index=False)
    df_pseudo.to_csv(OUT_DIR / "ctrl_fewlabel_pseudo.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
