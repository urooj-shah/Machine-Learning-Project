"""Control 4: Size analysis.

Per site, bin positive images by polyp area fraction (<1%, 1-3%, 3-10%, >10%).
Report n, source-only decoder Dice, and quantization ceiling per bin.
Report whether the site gap remains within matched bins.
Report share of polyps smaller than 2x2 patches.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import GRID, N_PATCHES, dice_score, load_gt_mask, load_site_features, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BINS = [(0.0, 0.01, "<1%"), (0.01, 0.03, "1-3%"), (0.03, 0.10, "3-10%"), (0.10, 1.01, ">10%")]
PATCH_AREA_FRAC = (14.0 / 224.0) ** 2  # one patch = (14/224)^2 of image
TWO_PATCH_AREA = 4 * PATCH_AREA_FRAC  # 2x2 patches


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
    import torch.nn as nn
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns_k = list(kvasir["filenames"])
    train_idx = np.array([fns_k.index(fn) for fn in split["train"]])
    val_idx = np.array([fns_k.index(fn) for fn in split["val"]])

    print("Caching masks ...", flush=True)
    kvasir_m128 = load_masks_128(kvasir["mask_paths"])

    print("Training decoder ...", flush=True)
    model, vd = train_decoder(
        kvasir["patch_tokens"][train_idx], kvasir_m128[train_idx],
        kvasir["patch_tokens"][val_idx], kvasir_m128[val_idx],
        n_epochs=80, seed=42)
    print(f"  val_dice={vd:.4f}", flush=True)

    target_sites = [f"polypgen_C{i}" for i in range(1, 7)]
    all_sites = [("kvasir_val", kvasir, val_idx)] + \
        [(nm, load_site_features(FEAT_DIR, nm), None) for nm in target_sites]

    rows = []
    small_polyp_rows = []

    for site_name, sd, indices in all_sites:
        if indices is None:
            indices = np.arange(len(sd["filenames"]))
        n = len(indices)
        tokens = sd["patch_tokens"][indices]
        mask_paths = sd["mask_paths"][indices]
        fg_fractions = sd["patch_fg_fraction"][indices]

        print(f"\n  {site_name} ({n}) ...", flush=True)

        # Decoder Dice per image
        logits = decoder_logits_128(model, tokens)
        probs = sigmoid(logits)

        # Per-image area fraction (from GT masks at original resolution)
        area_fracs = []
        gt_is_pos = []
        for i in range(n):
            gt = load_gt_mask(str(mask_paths[i]))
            af = gt.sum() / gt.size
            area_fracs.append(af)
            gt_is_pos.append(gt.sum() > 0)
        area_fracs = np.array(area_fracs)
        gt_is_pos = np.array(gt_is_pos)

        # Quantization ceiling per image: fg_fraction -> 16x16 binary -> upsample to 128 -> dice vs 128 GT
        gt_128_site = load_masks_128(mask_paths)
        for i in range(n):
            if not gt_is_pos[i]:
                continue
            # Patch-level binary GT (from cached patch_fg_fraction)
            patch_gt_16 = (fg_fractions[i].reshape(GRID, GRID) >= 0.5).astype(np.float32)
            quant_128 = np.array(Image.fromarray(patch_gt_16, mode="F")
                                 .resize((MASK_SIZE, MASK_SIZE), Image.NEAREST))
            quant_pred = (quant_128 >= 0.5).astype(np.uint8)
            gt_128_bin = (gt_128_site[i] >= 0.5).astype(np.uint8)
            q_dice = dice_score(quant_pred, gt_128_bin)

            # Decoder dice at 128
            pred_128 = (probs[i] >= 0.5).astype(np.uint8)
            d_dice = dice_score(pred_128, gt_128_bin)

            af = area_fracs[i]
            bin_label = "?"
            for lo, hi, label in BINS:
                if lo <= af < hi:
                    bin_label = label
                    break

            rows.append({
                "site": site_name, "img_idx": int(i),
                "area_frac": float(af), "bin": bin_label,
                "quant_ceiling": float(q_dice),
                "decoder_dice": float(d_dice),
            })

        # Share of polyps smaller than 2x2 patches
        pos_fracs = area_fracs[gt_is_pos]
        n_small = (pos_fracs < TWO_PATCH_AREA).sum()
        small_polyp_rows.append({
            "site": site_name,
            "n_pos": int(gt_is_pos.sum()),
            "n_below_2x2_patches": int(n_small),
            "pct_below_2x2": float(n_small / max(gt_is_pos.sum(), 1) * 100),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    # Aggregate by site + bin
    agg = df.groupby(["site", "bin"]).agg(
        n=("img_idx", "count"),
        quant_ceiling=("quant_ceiling", "mean"),
        decoder_dice=("decoder_dice", "mean"),
    ).reset_index()

    print("\n" + "=" * 100)
    print("TABLE: SIZE ANALYSIS (positive images, decoder seed 42, 128x128 eval)")
    print("=" * 100)
    print(agg.to_string(index=False))

    # Cross-site gap within matched bins
    print("\n" + "=" * 100)
    print("TABLE: CROSS-SITE GAP WITHIN MATCHED BINS")
    print("=" * 100)
    kv_agg = agg[agg["site"] == "kvasir_val"].set_index("bin")
    gap_rows = []
    for site in [s for s in agg["site"].unique() if s != "kvasir_val"]:
        site_agg = agg[agg["site"] == site].set_index("bin")
        for b in kv_agg.index:
            if b in site_agg.index:
                gap_rows.append({
                    "site": site, "bin": b,
                    "n_kvasir": int(kv_agg.loc[b, "n"]),
                    "n_target": int(site_agg.loc[b, "n"]),
                    "dice_kvasir": float(kv_agg.loc[b, "decoder_dice"]),
                    "dice_target": float(site_agg.loc[b, "decoder_dice"]),
                    "gap": float(kv_agg.loc[b, "decoder_dice"] - site_agg.loc[b, "decoder_dice"]),
                })
    df_gap = pd.DataFrame(gap_rows)
    print(df_gap.to_string(index=False))

    # Small polyps
    df_small = pd.DataFrame(small_polyp_rows)
    print("\n" + "=" * 80)
    print("TABLE: POLYPS SMALLER THAN 2x2 PATCHES")
    print("=" * 80)
    print(df_small.to_string(index=False))

    df.to_csv(OUT_DIR / "ctrl_size_per_image.csv", index=False)
    agg.to_csv(OUT_DIR / "ctrl_size_binned.csv", index=False)
    df_gap.to_csv(OUT_DIR / "ctrl_size_gap.csv", index=False)
    df_small.to_csv(OUT_DIR / "ctrl_size_small_polyps.csv", index=False)

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
