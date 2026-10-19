"""Control 2: Positive control — synthetic distribution shift.

Apply a known synthetic shift to Kvasir val features:
  per-channel scale ~ U(0.7, 1.4)
  per-channel offset ~ N(0, 0.5*std)
  random mixing matrix M = I + 0.2*R (R ~ N(0,1))

Report decoder Dice before and after the shift, then after each
alignment variant using the shifted set's own statistics.
Alignment must recover most of the loss.
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
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


def load_masks_128(mask_paths):
    masks = []
    for p in mask_paths:
        m = Image.open(str(p)).convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
        masks.append((np.array(m, dtype=np.float32) >= 128).astype(np.float32))
    return np.stack(masks)


def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def train_decoder(train_tokens, train_masks, val_tokens, val_masks,
                  n_epochs=80, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatureDataset(train_tokens, train_masks)
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
        vds = FeatureDataset(val_tokens, val_masks)
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


@torch.no_grad()
def decoder_logits_128(model, tokens):
    model.eval()
    out = []
    for i in range(0, len(tokens), 64):
        batch = tokens[i:i + 64]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_128(model, tokens, gt_128, threshold=0.5):
    logits = decoder_logits_128(model, tokens)
    probs = sigmoid(logits)
    dices = []
    for i in range(len(tokens)):
        pred = (probs[i] >= threshold).astype(np.uint8)
        gt = (gt_128[i] >= 0.5).astype(np.uint8)
        dices.append(dice_score(pred, gt))
    return np.array(dices)


def channel_standardize(X, mu_s, sigma_s, mu_t, sigma_t, eps=1e-6):
    return ((X - mu_t) / (sigma_t + eps) * sigma_s + mu_s).astype(np.float32)


def coral_stats(X):
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), np.cov(X, rowvar=False)


def coral_transform(X_target, mu_s, C_s, mu_t, C_t, reg=1e-5):
    D = C_s.shape[0]
    I = np.eye(D)
    d_t, V_t = np.linalg.eigh(C_t + reg * I)
    d_t = np.maximum(d_t, reg)
    Ct_isqrt = V_t @ np.diag(1.0 / np.sqrt(d_t)) @ V_t.T
    d_s, V_s = np.linalg.eigh(C_s + reg * I)
    d_s = np.maximum(d_s, reg)
    Cs_sqrt = V_s @ np.diag(np.sqrt(d_s)) @ V_s.T
    W = Ct_isqrt @ Cs_sqrt
    X = np.asarray(X_target, dtype=np.float64)
    return ((X - mu_t) @ W + mu_s).astype(np.float32)


@torch.no_grad()
def adapt_bn(model, target_tokens):
    adapted = copy.deepcopy(model)
    for m in adapted.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
            m.momentum = None
    adapted.train()
    for i in range(0, len(target_tokens), 64):
        batch = target_tokens[i:i + 64]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        adapted(x)
    adapted.eval()
    return adapted


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])

    print("Training decoder ...", flush=True)
    ts = time.time()
    model, vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=SEED)
    print(f"  val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)

    val_tokens = kvasir["patch_tokens"][val_idx]
    val_gt128 = kvasir_m128[val_idx]
    n_val = len(val_idx)

    # Source stats
    X_src = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    mu_s = X_src.mean(axis=0)
    sigma_s = X_src.std(axis=0)
    mu_s_cor, C_s_cor = coral_stats(X_src)
    lam = 1e-3 * np.diag(C_s_cor).mean()

    # Baseline: unshifted
    dices_clean = eval_128(model, val_tokens, val_gt128)
    print(f"\nClean Dice: {dices_clean.mean():.4f}", flush=True)

    # Generate synthetic shift components (fixed seed for reproducibility)
    rng = np.random.default_rng(SEED)
    scale = rng.uniform(0.7, 1.4, size=384).astype(np.float64)
    offset = rng.normal(0, 0.5, size=384).astype(np.float64) * sigma_s
    R = rng.standard_normal((384, 384)).astype(np.float64)
    M_full = np.eye(384) + 0.2 * R
    M_mild = np.eye(384) + 0.02 * R

    X_val = val_tokens.reshape(-1, 384).astype(np.float64)

    # Test decomposed shifts to diagnose where recovery breaks down
    shift_configs = [
        ("scale_offset_only", lambda x: (x * scale + offset), "per-channel scale+offset, no mixing"),
        ("mild_mixing", lambda x: (x * scale + offset) @ M_mild,
         f"scale+offset+mild mixing (cond={np.linalg.cond(M_mild):.1f})"),
        ("full_shift", lambda x: (x * scale + offset) @ M_full,
         f"scale+offset+full mixing (cond={np.linalg.cond(M_full):.1f})"),
    ]

    all_results = []

    for shift_name, shift_fn, desc in shift_configs:
        print(f"\n--- {shift_name}: {desc} ---", flush=True)
        X_shifted = shift_fn(X_val).astype(np.float32)
        tokens_shifted = X_shifted.reshape(n_val, N_PATCHES, 384)

        dices_shifted = eval_128(model, tokens_shifted, val_gt128)
        drop = dices_clean.mean() - dices_shifted.mean()
        print(f"  Shifted Dice: {dices_shifted.mean():.4f} (drop: {drop:.4f})")

        all_results.append({"shift": shift_name, "condition": "clean",
                            "dice": float(dices_clean.mean()), "drop": 0.0})
        all_results.append({"shift": shift_name, "condition": "shifted",
                            "dice": float(dices_shifted.mean()), "drop": float(drop)})

        if drop < 0.001:
            print("  Shift too small, skipping alignment")
            continue

        mu_t = X_shifted.reshape(-1, 384).astype(np.float64).mean(axis=0)
        sigma_t = X_shifted.reshape(-1, 384).astype(np.float64).std(axis=0)
        mu_t_cor, C_t_cor = coral_stats(X_shifted.reshape(-1, 384).astype(np.float64))

        for vname, get_tok_mdl in [
            ("B_channel_std", lambda: (channel_standardize(
                X_shifted.reshape(-1, 384), mu_s, sigma_s, mu_t, sigma_t
            ).reshape(n_val, N_PATCHES, 384), model)),
            ("C_coral", lambda: (coral_transform(
                X_shifted.reshape(-1, 384).astype(np.float64),
                mu_s_cor, C_s_cor, mu_t_cor, C_t_cor, reg=lam
            ).reshape(n_val, N_PATCHES, 384), model)),
            ("D_adabn", lambda: (tokens_shifted, adapt_bn(model, tokens_shifted))),
        ]:
            tok, mdl = get_tok_mdl()
            dices_v = eval_128(mdl, tok, val_gt128)
            recovery = (dices_v.mean() - dices_shifted.mean()) / drop
            all_results.append({"shift": shift_name, "condition": vname,
                                "dice": float(dices_v.mean()),
                                "recovery_pct": float(recovery * 100)})
            print(f"  {vname}: {dices_v.mean():.4f} (recovery: {recovery:.1%})")

    # Oracle: retrain on full shift
    print(f"\nOracle: retrain decoder on full-shifted training data ...", flush=True)
    X_train = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    X_shifted_full = ((X_train * scale + offset) @ M_full).astype(np.float32)
    train_tokens_shifted = X_shifted_full.reshape(len(train_idx), N_PATCHES, 384)
    X_val_full = ((X_val * scale + offset) @ M_full).astype(np.float32)
    tokens_shifted_full = X_val_full.reshape(n_val, N_PATCHES, 384)
    model_shifted, vd_shifted = train_decoder(
        train_tokens_shifted, kvasir_m128[train_idx],
        tokens_shifted_full, val_gt128,
        n_epochs=80, seed=SEED)
    dices_oracle = eval_128(model_shifted, tokens_shifted_full, val_gt128)
    drop_full = dices_clean.mean() - eval_128(model, tokens_shifted_full, val_gt128).mean()
    recovery_oracle = (dices_oracle.mean() - eval_128(model, tokens_shifted_full, val_gt128).mean()) / drop_full if drop_full > 0 else 0
    all_results.append({"shift": "full_shift", "condition": "oracle_retrained",
                        "dice": float(dices_oracle.mean()),
                        "recovery_pct": float(recovery_oracle * 100)})
    print(f"  Oracle retrained: {dices_oracle.mean():.4f} (recovery: {recovery_oracle:.1%})")

    results = all_results

    df = pd.DataFrame(results)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + "=" * 80)
    print("POSITIVE CONTROL: SYNTHETIC SHIFT RECOVERY")
    print("=" * 80)
    print(df.to_string(index=False))

    check_pass = any(r.get("recovery_pct", 0) > 50 for r in results if r["condition"] in ["B_channel_std", "C_coral", "D_adabn"])
    print(f"\n  At least one alignment recovers >50% of induced drop: {'YES' if check_pass else 'NO'}")

    df.to_csv(OUT_DIR / "ctrl_positive.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
