"""Quick PoC: Do DINOv2 attention maps highlight polyps?

Extract [CLS] attention from last ViT layer, threshold, compare with GT masks.
Report IoU and Dice of attention-based saliency vs ground truth per site.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 224
PATCH_SIZE = 14
GRID = SIZE // PATCH_SIZE  # 16


def get_image_mask_pairs(ds_cfg, max_images=None):
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
            pairs.append((img_dir / fname, mask_path))
    if max_images:
        pairs = pairs[:max_images]
    return pairs


def build_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model = model.to(DEVICE).eval()
    return model


def get_attention_maps(model, img_tensor):
    """Extract [CLS] attention to patches from last layer, all heads."""
    with torch.no_grad():
        out = model.get_intermediate_layers(img_tensor, n=1, return_class_token=True)

    # Use the forward hook approach to get attention weights
    attentions = []
    def hook_fn(module, input, output):
        # output is tuple, attention weights are computed inside
        pass

    # Alternative: use model.get_intermediate_layers with reshape=True
    # But we need raw attention. Let's register a hook on the last block's attn.
    attn_weights = {}
    def attn_hook(module, input, output):
        # For DINOv2, the attention module stores attn after softmax
        # We need to recompute: q, k from input
        B, N, C = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C // module.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn_weights["attn"] = attn.detach().cpu()

    last_block = model.blocks[-1]
    handle = last_block.attn.register_forward_hook(attn_hook)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()

    attn = attn_weights["attn"]  # [B, num_heads, N, N] where N = 1 + grid*grid
    cls_attn = attn[:, :, 0, 1:]  # [B, num_heads, grid*grid] — CLS to patches
    return cls_attn


def main():
    t0 = time.time()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = build_model()
    n_heads = model.blocks[-1].attn.num_heads

    all_sites = ["kvasir", "polypgen_C1", "polypgen_C2", "polypgen_C3",
                 "polypgen_C4", "polypgen_C5", "polypgen_C6"]

    print(f"{'site':>15} {'n':>5} {'attn_dice':>10} {'attn_iou':>10} "
          f"{'best_head':>10} {'best_dice':>11} {'pos_rate':>9}")
    print("-" * 80)

    for site_name in all_sites:
        pairs = get_image_mask_pairs(cfg["datasets"][site_name])
        dices_mean, ious_mean = [], []
        per_head_dices = [[] for _ in range(n_heads)]
        n_pos = 0

        for img_path, mask_path in pairs:
            gt = Image.open(str(mask_path)).convert("L").resize((GRID, GRID), Image.BILINEAR)
            gt_arr = (np.array(gt, dtype=np.float32) / 255.0)
            gt_binary = (gt_arr >= 0.5).astype(np.float32)

            if gt_binary.sum() == 0:
                continue
            n_pos += 1

            img = Image.open(str(img_path)).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)

            cls_attn = get_attention_maps(model, x)  # [1, heads, grid*grid]
            # Average across heads
            mean_attn = cls_attn[0].mean(dim=0).numpy()  # [grid*grid]
            mean_attn = mean_attn.reshape(GRID, GRID)

            # Normalize to [0,1]
            mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min() + 1e-8)

            # Threshold at 0.5 of normalized attention
            pred = (mean_attn >= 0.5).astype(np.float32)
            inter = (pred * gt_binary).sum()
            union = pred.sum() + gt_binary.sum() - inter
            dice = (2 * inter) / (pred.sum() + gt_binary.sum() + 1e-8)
            iou = inter / (union + 1e-8)
            dices_mean.append(dice)
            ious_mean.append(iou)

            # Per-head
            for h in range(n_heads):
                h_attn = cls_attn[0, h].numpy().reshape(GRID, GRID)
                h_attn = (h_attn - h_attn.min()) / (h_attn.max() - h_attn.min() + 1e-8)
                h_pred = (h_attn >= 0.5).astype(np.float32)
                h_dice = (2 * (h_pred * gt_binary).sum()) / (h_pred.sum() + gt_binary.sum() + 1e-8)
                per_head_dices[h].append(h_dice)

        head_means = [np.mean(d) for d in per_head_dices]
        best_h = int(np.argmax(head_means))

        print(f"{site_name:>15} {n_pos:>5} {np.mean(dices_mean):>10.4f} {np.mean(ious_mean):>10.4f} "
              f"{'head_' + str(best_h):>10} {head_means[best_h]:>11.4f} "
              f"{n_pos/len(pairs):>9.1%}")

    # Also test: use attention as weight for pseudo-labels
    print("\n\n=== Attention-weighted pseudo-label quality (Kvasir) ===")
    print("Compare: flat threshold vs attention-weighted threshold\n")

    # Load decoder for Kvasir
    sys.path.insert(0, str(ROOT / "src"))
    from eval import load_site_features, sigmoid, dice_score
    from patch_decoder import PatchDecoder
    import json

    kvasir = load_site_features(ROOT / "features" / "dinov2_vits14_224", "kvasir")
    with open(ROOT / "data" / "splits" / "kvasir_train_val.json") as f:
        split = json.load(f)
    fns = list(kvasir["filenames"])
    val_idx = np.array([fns.index(fn) for fn in split["val"]])

    ckpt = torch.load(ROOT / "checkpoints" / "decoder_corrected.pt", map_location=DEVICE, weights_only=True)
    decoder = PatchDecoder(in_dim=384).to(DEVICE)
    decoder.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    decoder.eval()

    # Get decoder predictions on val
    val_tokens = kvasir["patch_tokens"][val_idx]
    mask_paths = kvasir["mask_paths"][val_idx]

    n_compared = 0
    flat_dices, weighted_dices = [], []

    pairs_kvasir = get_image_mask_pairs(cfg["datasets"]["kvasir"])
    img_path_map = {}
    for ip, mp in pairs_kvasir:
        img_path_map[Path(mp).stem.replace("_mask", "")] = ip

    for i, idx in enumerate(val_idx):
        fn = fns[idx]
        if fn not in img_path_map:
            continue

        # GT at 128x128
        gt_128 = Image.open(str(mask_paths[i])).convert("L").resize((128, 128), Image.NEAREST)
        gt_128 = (np.array(gt_128, dtype=np.float32) >= 128).astype(np.float32)
        if gt_128.sum() == 0:
            continue

        # Decoder logits
        tok = val_tokens[i].reshape(GRID, GRID, -1).transpose(2, 0, 1)
        tok_t = torch.from_numpy(np.ascontiguousarray(tok)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = decoder(tok_t).cpu().numpy()[0, 0]  # [128, 128]
        probs = sigmoid(logits)

        # Attention map
        img = Image.open(str(img_path_map[fn])).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        cls_attn = get_attention_maps(model, x)
        mean_attn = cls_attn[0].mean(dim=0).numpy().reshape(GRID, GRID)
        # Upsample attention to 128x128
        attn_128 = np.array(Image.fromarray(mean_attn, mode="F").resize((128, 128), Image.BILINEAR))
        attn_128 = (attn_128 - attn_128.min()) / (attn_128.max() - attn_128.min() + 1e-8)

        # Flat pseudo-label: threshold probs at 0.5
        flat_pl = (probs >= 0.5).astype(np.float32)
        # Attention-weighted: threshold (probs * attn_weight) — boost where attention is high
        # Use attention to modulate confidence: effective_prob = probs * (0.5 + 0.5 * attn)
        weighted_prob = probs * (0.5 + 0.5 * attn_128)
        weighted_pl = (weighted_prob >= 0.5).astype(np.float32)

        flat_dices.append(dice_score(flat_pl.astype(np.uint8), gt_128.astype(np.uint8)))
        weighted_dices.append(dice_score(weighted_pl.astype(np.uint8), gt_128.astype(np.uint8)))
        n_compared += 1

    print(f"  n_images: {n_compared}")
    print(f"  Flat pseudo-label Dice vs GT:     {np.mean(flat_dices):.4f}")
    print(f"  Attn-weighted pseudo-label Dice:  {np.mean(weighted_dices):.4f}")
    print(f"  Delta: {np.mean(weighted_dices) - np.mean(flat_dices):+.4f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
