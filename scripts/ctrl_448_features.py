"""Control 5: Extract features at 448x448 (32x32 grid, same backbone).

Resumable: skips datasets whose output .npz already exists.
Then retrain decoder (5 seeds) and report per-site Dice, ceilings,
and extraction/training cost vs 224.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import GRID as GRID_224, dice_score, load_gt_mask, load_site_features, sigmoid

GRID_448 = 32
N_PATCHES_448 = GRID_448 * GRID_448

FEAT_DIR_224 = ROOT / "features" / "dinov2_vits14_224"
FEAT_DIR_448 = ROOT / "features" / "dinov2_vits14_448"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
DATA_ROOT = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs" / "08_controls"
MASK_SIZE_448 = 256  # decoder output for 32x32: 32->64->128->256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]

ALL_DATASETS = ["kvasir", "polypgen_C1", "polypgen_C2", "polypgen_C3",
                "polypgen_C4", "polypgen_C5", "polypgen_C6"]


def build_model(backbone="dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", backbone, verbose=False)
    model = model.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        model = model.half()
    return model


def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_image_mask_pairs(ds_cfg):
    img_dir = DATA_ROOT / ds_cfg["images"]
    mask_dir = DATA_ROOT / ds_cfg["masks"]
    pattern = ds_cfg["mask_pattern"]
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        stem = Path(fname).stem
        mask_name = pattern.format(stem=stem)
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            pairs.append((img_dir / fname, mask_path, stem))
    return pairs


def compute_patch_labels(mask_path, size, patch_size):
    mask = Image.open(mask_path).convert("L")
    orig_w, orig_h = mask.size
    mask = mask.resize((size, size), Image.NEAREST)
    m = np.array(mask, dtype=np.float32)
    m = (m >= 128).astype(np.float32)
    grid = size // patch_size
    fractions = np.zeros((grid, grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            patch = m[r * patch_size:(r + 1) * patch_size,
                      c * patch_size:(c + 1) * patch_size]
            fractions[r, c] = patch.mean()
    binary = (fractions >= 0.5).astype(np.int8)
    return fractions.ravel(), binary.ravel(), (orig_w, orig_h)


def extract_features(model_name, size, patch_size, ds_name, ds_cfg, out_dir):
    out_path = out_dir / f"{ds_name}.npz"
    if out_path.exists():
        print(f"    {ds_name}: already exists, skipping", flush=True)
        return out_path

    model = build_model(model_name)
    transform = build_transform(size)
    pairs = get_image_mask_pairs(ds_cfg)
    print(f"    {ds_name}: {len(pairs)} images", flush=True)

    all_tokens, all_fracs, all_binary = [], [], []
    filenames, mask_paths, orig_sizes = [], [], []

    for img_path, mask_path, stem in pairs:
        img = Image.open(str(img_path)).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        if DEVICE.type == "cuda":
            x = x.half()
        with torch.no_grad():
            out = model.forward_features(x)
            tokens = out["x_norm_patchtokens"].cpu().float().numpy()[0]
        all_tokens.append(tokens.astype(np.float16))

        frac, binary, (ow, oh) = compute_patch_labels(mask_path, size, patch_size)
        all_fracs.append(frac)
        all_binary.append(binary)
        filenames.append(stem)
        mask_paths.append(str(mask_path))
        orig_sizes.append((ow, oh))

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_dir), suffix=".npz")
    os.close(tmp_fd)
    np.savez_compressed(
        tmp_path,
        patch_tokens=np.stack(all_tokens),
        patch_fg_fraction=np.stack(all_fracs),
        patch_fg_binary=np.stack(all_binary),
        filenames=np.array(filenames),
        mask_paths=np.array(mask_paths),
        orig_sizes=np.array(orig_sizes),
    )
    os.rename(tmp_path, str(out_path))
    del model
    torch.cuda.empty_cache()
    return out_path


def load_site_features_448(feat_dir, name):
    d = np.load(feat_dir / f"{name}.npz")
    return {
        "patch_tokens": d["patch_tokens"].astype(np.float32),
        "patch_fg_binary": d["patch_fg_binary"],
        "patch_fg_fraction": d["patch_fg_fraction"].astype(np.float32),
        "filenames": d["filenames"],
        "mask_paths": d["mask_paths"],
        "orig_sizes": d["orig_sizes"],
    }


class FeatureDataset448(Dataset):
    def __init__(self, patch_tokens, masks):
        self.tokens = patch_tokens
        self.masks = masks
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        x = self.tokens[idx].reshape(GRID_448, GRID_448, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), \
               torch.from_numpy(self.masks[idx]).unsqueeze(0)


def load_masks(mask_paths, size):
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


from patch_decoder import PatchDecoder


def train_decoder_448(train_tokens, train_masks, val_tokens, val_masks,
                      n_epochs=80, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FeatureDataset448(train_tokens, train_masks)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
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
        vds = FeatureDataset448(val_tokens, val_masks)
        vdl = DataLoader(vds, batch_size=16, shuffle=False, num_workers=0)
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
def decoder_logits(model, tokens, grid):
    model.eval()
    out = []
    bs = 32 if grid == 16 else 16
    for i in range(0, len(tokens), bs):
        batch = tokens[i:i + bs]
        x = np.stack([t.reshape(grid, grid, -1).transpose(2, 0, 1) for t in batch])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def eval_dice(model, tokens, gt_masks, grid, mask_size):
    logits = decoder_logits(model, tokens, grid)
    probs = sigmoid(logits)
    dices = []
    for i in range(len(tokens)):
        pred = (probs[i] >= 0.5).astype(np.uint8)
        gt = (gt_masks[i] >= 0.5).astype(np.uint8)
        dices.append(dice_score(pred, gt))
    return np.array(dices)


def quant_ceiling(fg_fraction, gt_masks, grid, mask_size):
    dices = []
    for i in range(len(fg_fraction)):
        patch_gt = (fg_fraction[i].reshape(grid, grid) >= 0.5).astype(np.float32)
        quant = np.array(Image.fromarray(patch_gt, mode="F")
                         .resize((mask_size, mask_size), Image.NEAREST))
        pred = (quant >= 0.5).astype(np.uint8)
        gt = (gt_masks[i] >= 0.5).astype(np.uint8)
        if gt.sum() > 0:
            dices.append(dice_score(pred, gt))
    return np.array(dices)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR_448.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Step 1: Extract features at 448
    print("=== Step 1: Feature extraction at 448x448 ===", flush=True)
    t_extract = time.time()
    for ds_name in ALL_DATASETS:
        extract_features("dinov2_vits14", 448, 14, ds_name, cfg["datasets"][ds_name], FEAT_DIR_448)
    t_extract = time.time() - t_extract
    print(f"  Extraction time: {t_extract:.0f}s", flush=True)

    # Step 2: Train 5 decoders at 448
    print("\n=== Step 2: Train decoders (448) ===", flush=True)
    kvasir_448 = load_site_features_448(FEAT_DIR_448, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir_448["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    masks_448_train = load_masks(kvasir_448["mask_paths"][train_idx], MASK_SIZE_448)
    masks_448_val = load_masks(kvasir_448["mask_paths"][val_idx], MASK_SIZE_448)

    decoders_448 = []
    vd_448s = []
    t_train_448 = time.time()
    for seed in SEEDS:
        ts = time.time()
        model, vd = train_decoder_448(
            kvasir_448["patch_tokens"][train_idx], masks_448_train,
            kvasir_448["patch_tokens"][val_idx], masks_448_val,
            n_epochs=80, seed=seed)
        decoders_448.append(model)
        vd_448s.append(vd)
        print(f"  Seed {seed}: val_dice={vd:.4f} ({time.time() - ts:.0f}s)", flush=True)
    t_train_448 = time.time() - t_train_448
    print(f"  Mean: {np.mean(vd_448s):.4f} +- {np.std(vd_448s):.4f}")

    # Step 3: Also train 224 decoders for comparison
    print("\n=== Step 3: Train decoders (224) ===", flush=True)
    kvasir_224 = load_site_features(FEAT_DIR_224, "kvasir")
    masks_224_train = load_masks(kvasir_224["mask_paths"][train_idx], 128)
    masks_224_val = load_masks(kvasir_224["mask_paths"][val_idx], 128)

    decoders_224 = []
    vd_224s = []
    t_train_224 = time.time()
    for seed in SEEDS:
        ts = time.time()
        from torch.utils.data import DataLoader as DL2

        class FD224(Dataset):
            def __init__(self, tokens, masks):
                self.tokens = tokens
                self.masks = masks
            def __len__(self):
                return len(self.tokens)
            def __getitem__(self, idx):
                x = self.tokens[idx].reshape(16, 16, -1).transpose(2, 0, 1)
                return torch.from_numpy(np.ascontiguousarray(x)), \
                       torch.from_numpy(self.masks[idx]).unsqueeze(0)

        torch.manual_seed(seed)
        np.random.seed(seed)
        ds = FD224(kvasir_224["patch_tokens"][train_idx], masks_224_train)
        dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
        model224 = PatchDecoder(in_dim=384).to(DEVICE)
        opt = torch.optim.Adam(model224.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        bce = nn.BCEWithLogitsLoss()
        best_dice, best_state = 0.0, None
        for _ in range(80):
            model224.train()
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = bce(model224(x), y) + dice_loss(model224(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            model224.eval()
            vds = FD224(kvasir_224["patch_tokens"][val_idx], masks_224_val)
            vdl = DataLoader(vds, batch_size=32, shuffle=False, num_workers=0)
            td = 0
            with torch.no_grad():
                for x, y in vdl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    p = (torch.sigmoid(model224(x)) >= 0.5).float()
                    inter = (p * y).sum(dim=(1, 2, 3))
                    union = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                    td += ((2 * inter + 1) / (union + 1)).sum().item()
            vdd = td / len(vds)
            if vdd > best_dice:
                best_dice = vdd
                best_state = {k: v.cpu().clone() for k, v in model224.state_dict().items()}
        if best_state:
            model224.load_state_dict(best_state)
        decoders_224.append(model224)
        vd_224s.append(best_dice)
        print(f"  Seed {seed}: val_dice={best_dice:.4f} ({time.time() - ts:.0f}s)", flush=True)
    t_train_224 = time.time() - t_train_224
    print(f"  Mean: {np.mean(vd_224s):.4f} +- {np.std(vd_224s):.4f}")

    # Step 4: Evaluate both on all sites
    print("\n=== Step 4: Evaluate ===", flush=True)
    target_sites = [f"polypgen_C{i}" for i in range(1, 7)]
    rows = []

    for site_name in ["kvasir_val"] + target_sites:
        if site_name == "kvasir_val":
            sd224 = kvasir_224
            sd448 = kvasir_448
            idx = val_idx
        else:
            sd224 = load_site_features(FEAT_DIR_224, site_name)
            sd448 = load_site_features_448(FEAT_DIR_448, site_name)
            idx = np.arange(len(sd224["filenames"]))

        n = len(idx)
        m224 = load_masks(sd224["mask_paths"][idx], 128)
        m448 = load_masks(sd448["mask_paths"][idx], MASK_SIZE_448)
        pos_mask_224 = np.array([(m >= 0.5).sum() > 0 for m in m224])

        # 224 eval
        dice_224_seeds = []
        for mdl in decoders_224:
            d = eval_dice(mdl, sd224["patch_tokens"][idx], m224, 16, 128)
            dice_224_seeds.append(d[pos_mask_224].mean() if pos_mask_224.sum() > 0 else float("nan"))
        # 448 eval
        dice_448_seeds = []
        for mdl in decoders_448:
            d = eval_dice(mdl, sd448["patch_tokens"][idx], m448, GRID_448, MASK_SIZE_448)
            pos_mask_448 = np.array([(m >= 0.5).sum() > 0 for m in m448])
            dice_448_seeds.append(d[pos_mask_448].mean() if pos_mask_448.sum() > 0 else float("nan"))

        # Ceilings
        ceil_224 = quant_ceiling(sd224["patch_fg_fraction"][idx], m224, 16, 128)
        ceil_448 = quant_ceiling(sd448["patch_fg_fraction"][idx], m448, GRID_448, MASK_SIZE_448)

        rows.append({
            "site": site_name,
            "dice_224_mean": float(np.nanmean(dice_224_seeds)),
            "dice_224_std": float(np.nanstd(dice_224_seeds)),
            "dice_448_mean": float(np.nanmean(dice_448_seeds)),
            "dice_448_std": float(np.nanstd(dice_448_seeds)),
            "ceiling_224": float(ceil_224.mean()) if len(ceil_224) > 0 else float("nan"),
            "ceiling_448": float(ceil_448.mean()) if len(ceil_448) > 0 else float("nan"),
        })
        print(f"  {site_name}: 224={np.nanmean(dice_224_seeds):.4f} 448={np.nanmean(dice_448_seeds):.4f} "
              f"ceil_224={ceil_224.mean():.4f} ceil_448={ceil_448.mean():.4f}", flush=True)

    # Cost
    rows_cost = [{
        "resolution": "224", "extraction_s": "cached", "training_s": f"{t_train_224:.0f}",
        "val_dice_mean": f"{np.mean(vd_224s):.4f}",
    }, {
        "resolution": "448", "extraction_s": f"{t_extract:.0f}", "training_s": f"{t_train_448:.0f}",
        "val_dice_mean": f"{np.mean(vd_448s):.4f}",
    }]

    df = pd.DataFrame(rows)
    df_cost = pd.DataFrame(rows_cost)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 140)
    print("TABLE: 224 vs 448 RESOLUTION (decoder, 5 seeds, native eval)")
    print("=" * 140)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE: COST")
    print("=" * 80)
    print(df_cost.to_string(index=False))

    df.to_csv(OUT_DIR / "ctrl_448_comparison.csv", index=False)
    df_cost.to_csv(OUT_DIR / "ctrl_448_cost.csv", index=False)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
