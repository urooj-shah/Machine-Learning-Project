"""Reproduce three anchor results from cached features and saved checkpoint."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import GRID, N_PATCHES, dice_score, load_gt_mask, load_site_features, upsample_probs, sigmoid
from patch_decoder import PatchDecoder

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"
CKPT_PATH = ROOT / "checkpoints" / "decoder_corrected.pt"
MASK_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantization_ceiling_kvasir_val():
    kvasir = load_site_features(FEAT_DIR, "kvasir")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    val_idx = [fns.index(fn) for fn in split["val"]]

    dices = []
    for idx in val_idx:
        gt = load_gt_mask(str(kvasir["mask_paths"][idx]))
        fg = kvasir["patch_fg_binary"][idx].reshape(GRID, GRID).astype(np.float32)
        ow, oh = int(kvasir["orig_sizes"][idx][0]), int(kvasir["orig_sizes"][idx][1])
        pred_up = upsample_probs(fg, ow, oh)
        pred = (pred_up >= 0.5).astype(np.uint8)
        dices.append(dice_score(pred, gt))
    return float(np.mean(dices))


@torch.no_grad()
def decoder_dice(site_name, model, feat_dir, split_path=None):
    sd = load_site_features(feat_dir, site_name)
    if split_path:
        with open(split_path) as f:
            split = json.load(f)
        fns = list(sd["filenames"])
        indices = [fns.index(fn) for fn in split["val"]]
    else:
        indices = list(range(len(sd["filenames"])))

    model.eval()
    dices = []
    for idx in indices:
        tokens = sd["patch_tokens"][idx].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(tokens)).unsqueeze(0).to(DEVICE)
        logits = model(x).cpu().numpy()[0, 0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        prob_f32 = probs.astype(np.float32)
        ow, oh = int(sd["orig_sizes"][idx][0]), int(sd["orig_sizes"][idx][1])
        img = Image.fromarray(prob_f32, mode="F")
        prob_up = np.array(img.resize((ow, oh), Image.BILINEAR))
        pred = (prob_up >= 0.5).astype(np.uint8)
        gt = load_gt_mask(str(sd["mask_paths"][idx]))
        dices.append(dice_score(pred, gt))
    return float(np.mean(dices))


def main():
    print("=" * 60)
    print("ANCHOR REPRODUCTION")
    print("=" * 60)

    ceiling = quantization_ceiling_kvasir_val()
    print(f"  Quantization ceiling (Kvasir val): {ceiling:.4f}  (expected ~0.916)")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = PatchDecoder(in_dim=384).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    pi_source = ckpt.get("pi_source", None)
    print(f"  Decoder checkpoint: epoch={ckpt.get('epoch')}, val_dice={ckpt.get('val_dice', '?')}, pi_source={pi_source}")

    kvasir_dice = decoder_dice("kvasir", model, FEAT_DIR, SPLIT_PATH)
    print(f"  Decoder Kvasir-val Dice: {kvasir_dice:.4f}  (expected ~0.840)")

    c4_dice = decoder_dice("polypgen_C4", model, FEAT_DIR)
    print(f"  Decoder C4 Dice:         {c4_dice:.4f}  (expected ~0.43)")

    ok = True
    for name, val, expected in [
        ("Quantization ceiling", ceiling, 0.916),
        ("Decoder Kvasir-val", kvasir_dice, 0.840),
        ("Decoder C4", c4_dice, 0.43),
    ]:
        diff = abs(val - expected)
        tol = 0.015 if name == "Decoder C4" else 0.01
        status = "OK" if diff < tol else f"DISCREPANCY ({diff:.4f})"
        print(f"  {name}: {status}")
        if diff >= 0.01:
            ok = False

    if not ok:
        print("\n  *** ANCHOR MISMATCH — debug before continuing ***")
        sys.exit(1)
    else:
        print("\n  All anchors reproduced within tolerance.")


if __name__ == "__main__":
    main()
