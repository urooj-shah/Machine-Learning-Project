"""Few-label ablation: 10/50/200 labels × 5 seeds, LR + decoder.

Task 7 from audit checklist. Evaluates on kvasir_val.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import (
    GRID, N_PATCHES, dice_score, load_gt_mask, load_site_features,
    sigmoid, upsample_probs,
)
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
OUT_DIR = ROOT / "outputs" / "05_audit"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureDataset(Dataset):
    def __init__(self, tokens, masks):
        self.tokens = tokens
        self.masks = masks
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
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2*inter+1)/(union+1)).mean()


def train_decoder(train_tok, train_m, val_tok, val_m, n_epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = FeatureDataset(train_tok, train_m)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model = PatchDecoder(in_dim=384).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss()
    best_d, best_s = 0.0, None
    for _ in range(n_epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        vds = FeatureDataset(val_tok, val_m)
        vdl = DataLoader(vds, batch_size=32, shuffle=False, num_workers=0)
        td = 0
        with torch.no_grad():
            for x, y in vdl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                inter = (p*y).sum(dim=(1,2,3))
                union = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
                td += ((2*inter+1)/(union+1)).sum().item()
        vd = td / len(vds)
        if vd > best_d:
            best_d = vd
            best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_s:
        model.load_state_dict(best_s)
    return model, best_d


@torch.no_grad()
def dec_logits(model, tokens):
    model.eval()
    out = []
    for i in range(0, len(tokens), 32):
        b = tokens[i:i+32]
        x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in b])
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
        out.append(model(x).cpu().numpy()[:, 0])
    return np.concatenate(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data ...", flush=True)
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    train_idx = np.array([fns.index(fn) for fn in split["train"]])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    m128 = load_masks_128(kvasir["mask_paths"])
    val_gt = [load_gt_mask(str(kvasir["mask_paths"][idx])) for idx in val_idx]

    print("\n" + "="*80)
    print("TASK 7: FEW-LABEL ABLATION (kvasir val, mean Dice)")
    print("="*80)

    abl = []
    t0 = time.time()
    for nl in [10, 50, 200]:
        for seed in [42, 43, 44, 45, 46]:
            rng = np.random.default_rng(seed)
            si = rng.choice(train_idx, size=nl, replace=False)

            # LR
            Xsub = kvasir["patch_tokens"][si].reshape(-1, 384)
            ysub = kvasir["patch_fg_binary"][si].reshape(-1)
            clfs = LogisticRegression(max_iter=1000, random_state=seed,
                                      solver="lbfgs", class_weight="balanced")
            clfs.fit(Xsub, ysub)
            Xv = kvasir["patch_tokens"][val_idx].reshape(-1, 384)
            lr_pv = clfs.predict_proba(Xv)[:, 1].astype(np.float32)
            lr_d = []
            for j, idx in enumerate(val_idx):
                ow, oh = int(kvasir["orig_sizes"][idx][0]), int(kvasir["orig_sizes"][idx][1])
                g = lr_pv[j*N_PATCHES:(j+1)*N_PATCHES].reshape(GRID, GRID)
                pm = upsample_probs(g, ow, oh)
                lr_d.append(dice_score((pm >= 0.5).astype(np.uint8), val_gt[j]))

            # Decoder
            decs, dvd = train_decoder(
                kvasir["patch_tokens"][si], m128[si],
                kvasir["patch_tokens"][val_idx], m128[val_idx],
                n_epochs=80, seed=seed)
            dl_v = dec_logits(decs, kvasir["patch_tokens"][val_idx])
            dp_v = sigmoid(dl_v)
            dec_d = []
            for j, idx in enumerate(val_idx):
                ow, oh = int(kvasir["orig_sizes"][idx][0]), int(kvasir["orig_sizes"][idx][1])
                a = np.asarray(dp_v[j], dtype=np.float32)
                img = Image.fromarray(a, mode="F")
                pm = np.array(img.resize((int(ow), int(oh)), Image.BILINEAR))
                dec_d.append(dice_score((pm >= 0.5).astype(np.uint8), val_gt[j]))

            abl.append({"n_labels": nl, "seed": seed,
                        "lr_dice": float(np.mean(lr_d)),
                        "dec_dice": float(np.mean(dec_d))})
            print(f"  n={nl:>3d}, seed={seed}: "
                  f"LR={np.mean(lr_d):.4f}  Dec={np.mean(dec_d):.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    pd.set_option("display.float_format", "{:.4f}".format)
    df = pd.DataFrame(abl)
    print("\n  TABLE 7: PER-SEED RESULTS")
    print(df.to_string(index=False))

    agg = df.groupby("n_labels").agg(
        lr_mean=("lr_dice", "mean"), lr_std=("lr_dice", "std"),
        dec_mean=("dec_dice", "mean"), dec_std=("dec_dice", "std"),
    ).reset_index()
    print("\n  AGGREGATED (mean ± std across seeds):")
    print(agg.to_string(index=False))

    df.to_csv(OUT_DIR / "fewlabel_ablation.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/fewlabel_ablation.csv")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
