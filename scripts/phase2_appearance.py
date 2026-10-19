"""Phase 2: Label-free appearance alignment proof of concept.

Evaluates at 128x128 resolution for speed. Trains decoder with 5 seeds.
Variants: A (none), B (channel std), C (CORAL), D (AdaBN).
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
from eval import (
    GRID, N_PATCHES, bootstrap_ci, dice_score,
    load_site_features, sigmoid,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "06_poc"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]

TARGET_SITES = ["polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


class FeatureDataset(Dataset):
    def __init__(self, patch_tokens, masks_128):
        self.tokens = patch_tokens
        self.masks = masks_128

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        x = np.ascontiguousarray(x)
        y = self.masks[idx]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


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
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
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
    """Evaluate at 128x128 resolution. Returns per-image dicts."""
    logits = decoder_logits_128(model, tokens)
    probs = sigmoid(logits)
    rows = []
    for i in range(len(tokens)):
        pred = (probs[i] >= threshold).astype(np.uint8)
        gt = (gt_128[i] >= 0.5).astype(np.uint8)
        pb, gb = pred.astype(bool), gt.astype(bool)
        tp = int((pb & gb).sum())
        fp = int((pb & ~gb).sum())
        fn = int((~pb & gb).sum())
        is_empty = (gb.sum() == 0)
        rows.append({
            "dice": dice_score(pred, gt),
            "is_empty": is_empty,
            "fp_on_empty": bool(is_empty and pb.sum() > 0),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def summarize(res_list):
    dices = np.array([r["dice"] for r in res_list])
    pos_dices = np.array([r["dice"] for r in res_list if not r["is_empty"]])
    empty = [r for r in res_list if r["is_empty"]]
    fp_e = [r["fp_on_empty"] for r in empty] if empty else []
    tp_s = sum(r["tp"] for r in res_list)
    fp_s = sum(r["fp"] for r in res_list)
    fn_s = sum(r["fn"] for r in res_list)
    pooled = 2 * tp_s / max(2 * tp_s + fp_s + fn_s, 1)
    return {
        "dice": float(dices.mean()),
        "pooled": pooled,
        "dice_pos": float(pos_dices.mean()) if len(pos_dices) > 0 else float("nan"),
        "n_pos": len(pos_dices),
        "n_empty": len(empty),
        "fp_rate_empty": float(np.mean(fp_e)) if fp_e else 0.0,
    }


# ── Alignment methods ────────────────────────────────────────────

def channel_standardize(X, mu_s, sigma_s, mu_t, sigma_t, eps=1e-6):
    return ((X - mu_t) / (sigma_t + eps) * sigma_s + mu_s).astype(np.float32)


def coral_stats(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    return mu, C


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

    targets = {}
    for name in TARGET_SITES:
        targets[name] = load_site_features(FEAT_DIR, name)

    # Cache all GT masks at 128x128
    print("Caching masks at 128x128 ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])
    target_m128 = {}
    for name in TARGET_SITES:
        target_m128[name] = load_masks_128(targets[name]["mask_paths"])

    # Source statistics
    print("Computing source statistics ...", flush=True)
    X_src = kvasir["patch_tokens"][train_idx].reshape(-1, 384).astype(np.float64)
    mu_s_ch = X_src.mean(axis=0)
    sigma_s_ch = X_src.std(axis=0)
    mu_s_coral, C_s_coral = coral_stats(X_src)
    lam_coral = 1e-3 * np.diag(C_s_coral).mean()
    print(f"  CORAL lambda: {lam_coral:.6f}")

    # Train 5 decoders
    print(f"\nTraining {len(SEEDS)} decoders ...", flush=True)
    decoders = []
    val_dices = []
    for seed in SEEDS:
        ts = time.time()
        model, vd = train_decoder(
            kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
            kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
            n_epochs=80, seed=seed)
        decoders.append(model)
        val_dices.append(vd)
        print(f"  Seed {seed}: val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)
    print(f"  Mean: {np.mean(val_dices):.4f} +- {np.std(val_dices):.4f}")

    # Evaluate
    sites = [("kvasir_val", kvasir, val_idx, kvasir_m128[val_idx])]
    for nm in TARGET_SITES:
        n_t = len(targets[nm]["filenames"])
        sites.append((nm, targets[nm], np.arange(n_t), target_m128[nm]))

    variants = ["A_none", "B_channel_std", "C_coral", "D_adabn"]
    all_rows = []
    cost_rows = []

    for site_name, sd, indices, gt_128 in sites:
        n = len(indices)
        print(f"\n  {site_name} ({n}) ...", flush=True)

        tokens_orig = sd["patch_tokens"][indices]

        # Target statistics
        X_tgt = tokens_orig.reshape(-1, 384).astype(np.float64)
        mu_t_ch = X_tgt.mean(axis=0)
        sigma_t_ch = X_tgt.std(axis=0)

        # Pre-compute aligned tokens
        t_ch = time.time()
        tokens_B = channel_standardize(
            tokens_orig.reshape(-1, 384), mu_s_ch, sigma_s_ch, mu_t_ch, sigma_t_ch
        ).reshape(n, N_PATCHES, 384)
        t_ch = time.time() - t_ch

        t_cor = time.time()
        mu_t_coral, C_t_coral = coral_stats(X_tgt)
        tokens_C = coral_transform(
            X_tgt, mu_s_coral, C_s_coral, mu_t_coral, C_t_coral, reg=lam_coral
        ).reshape(n, N_PATCHES, 384)
        t_cor = time.time() - t_cor

        t_abn = 0.0

        for variant in variants:
            seed_dices = []
            seed_dices_pos = []
            seed_fp_rates = []

            for si, (seed, model) in enumerate(zip(SEEDS, decoders)):
                if variant == "A_none":
                    tok, cur_model = tokens_orig, model
                elif variant == "B_channel_std":
                    tok, cur_model = tokens_B, model
                elif variant == "C_coral":
                    tok, cur_model = tokens_C, model
                elif variant == "D_adabn":
                    tok = tokens_orig
                    ta = time.time()
                    cur_model = adapt_bn(model, tokens_orig)
                    if si == 0:
                        t_abn = time.time() - ta

                res = eval_128(cur_model, tok, gt_128)
                m = summarize(res)
                seed_dices.append(m["dice"])
                seed_dices_pos.append(m["dice_pos"])
                seed_fp_rates.append(m["fp_rate_empty"])

            all_rows.append({
                "variant": variant, "site": site_name, "n": n,
                "dice_mean": float(np.mean(seed_dices)),
                "dice_std": float(np.std(seed_dices)),
                "dice_pos_mean": float(np.nanmean(seed_dices_pos)),
                "dice_pos_std": float(np.nanstd(seed_dices_pos)),
                "fp_rate_empty": float(np.mean(seed_fp_rates)),
                "n_pos": m["n_pos"], "n_empty": m["n_empty"],
            })

        # Paired bootstrap CIs (seed 0)
        res_A = eval_128(decoders[0], tokens_orig, gt_128)
        dices_A = np.array([r["dice"] for r in res_A])
        pos_mask = np.array([not r["is_empty"] for r in res_A])

        for vname, tok, mdl in [
            ("B_channel_std", tokens_B, decoders[0]),
            ("C_coral", tokens_C, decoders[0]),
            ("D_adabn", tokens_orig, adapt_bn(decoders[0], tokens_orig)),
        ]:
            res_v = eval_128(mdl, tok, gt_128)
            dices_v = np.array([r["dice"] for r in res_v])
            deltas = dices_v - dices_A
            ci = bootstrap_ci(deltas)

            row = next(r for r in all_rows if r["variant"] == vname and r["site"] == site_name)
            row["delta_seed0"] = float(deltas.mean())
            row["CI_lo"] = ci[0]
            row["CI_hi"] = ci[1]

            if pos_mask.sum() > 0:
                dp = dices_v[pos_mask] - dices_A[pos_mask]
                ci_p = bootstrap_ci(dp)
                row["delta_pos_seed0"] = float(dp.mean())
                row["CI_pos_lo"] = ci_p[0]
                row["CI_pos_hi"] = ci_p[1]

        if site_name.startswith("polypgen"):
            cost_rows.append({
                "site": site_name, "n_images": n,
                "t_channel_std_s": round(t_ch, 3),
                "t_coral_s": round(t_cor, 3),
                "t_adabn_s": round(t_abn, 3),
                "extra_params": 0,
            })

    # Ablation: n_unlabeled for best variant
    print("\n  N_unlabeled ablation ...", flush=True)
    ablation_rows = []
    df_tmp = pd.DataFrame(all_rows)
    tgt = df_tmp[df_tmp["site"].str.startswith("polypgen") & (df_tmp["variant"] != "A_none")]
    if "delta_seed0" in tgt.columns:
        best_variant = tgt.dropna(subset=["delta_seed0"]).groupby("variant")["delta_seed0"].mean().idxmax()
    else:
        best_variant = "B_channel_std"
    print(f"  Best variant: {best_variant}")

    model0 = decoders[0]
    for site_name, sd, indices, gt_128 in sites:
        if site_name == "kvasir_val":
            continue
        n = len(indices)
        tokens_orig = sd["patch_tokens"][indices]

        for n_unl in [8, 32, 128]:
            if n_unl >= n:
                continue
            rng = np.random.default_rng(42)
            sub = rng.choice(n, size=n_unl, replace=False)

            if best_variant == "B_channel_std":
                X_sub = tokens_orig[sub].reshape(-1, 384).astype(np.float64)
                mu_sub, sigma_sub = X_sub.mean(axis=0), X_sub.std(axis=0)
                tok_a = channel_standardize(
                    tokens_orig.reshape(-1, 384), mu_s_ch, sigma_s_ch, mu_sub, sigma_sub
                ).reshape(n, N_PATCHES, 384)
                cur_model = model0
            elif best_variant == "C_coral":
                X_sub = tokens_orig[sub].reshape(-1, 384).astype(np.float64)
                mu_sub, C_sub = coral_stats(X_sub)
                tok_a = coral_transform(
                    tokens_orig.reshape(-1, 384).astype(np.float64),
                    mu_s_coral, C_s_coral, mu_sub, C_sub, reg=lam_coral
                ).reshape(n, N_PATCHES, 384)
                cur_model = model0
            elif best_variant == "D_adabn":
                tok_a = tokens_orig
                cur_model = adapt_bn(model0, tokens_orig[sub])
            else:
                continue

            res = eval_128(cur_model, tok_a, gt_128)
            m = summarize(res)
            ablation_rows.append({
                "variant": best_variant, "site": site_name,
                "n_unlabeled": n_unl, "n_total": n,
                "dice": m["dice"], "dice_pos": m["dice_pos"],
            })

        # "all" point
        row = next(r for r in all_rows if r["variant"] == best_variant and r["site"] == site_name)
        ablation_rows.append({
            "variant": best_variant, "site": site_name,
            "n_unlabeled": n, "n_total": n,
            "dice": row["dice_mean"], "dice_pos": row["dice_pos_mean"],
        })

    # Print
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    df = pd.DataFrame(all_rows)
    cols = ["variant", "site", "n", "dice_mean", "dice_std",
            "dice_pos_mean", "dice_pos_std", "fp_rate_empty"]
    print("\n" + "=" * 140)
    print("TABLE: APPEARANCE ALIGNMENT (decoder, 5 seeds, 128x128 eval)")
    print("=" * 140)
    print(df[cols].to_string(index=False))

    ci_cols = ["variant", "site", "delta_seed0", "CI_lo", "CI_hi",
               "delta_pos_seed0", "CI_pos_lo", "CI_pos_hi"]
    ci_present = [c for c in ci_cols if c in df.columns]
    if len(ci_present) > 2:
        print("\n" + "=" * 120)
        print("PAIRED BOOTSTRAP CIs (seed 0)")
        print("=" * 120)
        print(df[df["variant"] != "A_none"][ci_present].to_string(index=False))

    if ablation_rows:
        df_abl = pd.DataFrame(ablation_rows)
        print("\n" + "=" * 100)
        print(f"N_UNLABELED ABLATION ({best_variant})")
        print("=" * 100)
        print(df_abl.to_string(index=False))
        df_abl.to_csv(OUT_DIR / "phase2_ablation.csv", index=False)

    if cost_rows:
        df_cost = pd.DataFrame(cost_rows)
        print("\n" + "=" * 80)
        print("ADAPTATION COST")
        print("=" * 80)
        print(df_cost.to_string(index=False))
        df_cost.to_csv(OUT_DIR / "phase2_cost.csv", index=False)

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    for variant in ["B_channel_std", "C_coral", "D_adabn"]:
        wins = 0
        kv_drop = 0.0
        for _, row in df.iterrows():
            if row["variant"] == variant:
                if row["site"] == "kvasir_val":
                    base_kv = df[(df["variant"] == "A_none") & (df["site"] == "kvasir_val")]
                    kv_drop = base_kv.iloc[0]["dice_pos_mean"] - row["dice_pos_mean"]
                elif row["site"].startswith("polypgen"):
                    if "CI_pos_lo" in row and pd.notna(row.get("CI_pos_lo")) and row["CI_pos_lo"] > 0:
                        wins += 1
        meets = wins >= 3 and kv_drop <= 0.01
        print(f"  {variant}: {wins}/6 wins, Kvasir drop={kv_drop:.4f}, MEETS={'YES' if meets else 'NO'}")

    df.to_csv(OUT_DIR / "phase2_alignment.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
