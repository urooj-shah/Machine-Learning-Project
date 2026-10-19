"""Extract frozen DINOv2 features and patch-level mask labels.

Resumable: skips datasets whose output .npz already exists.
Writes to a temp file then renames for crash safety.

NOTE: images are resized to size x size (square), which distorts the
aspect ratio. This is a known simplification — acceptable for patch-level
classification but not ideal for pixel-perfect segmentation.
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_image_mask_pairs(cfg, data_root):
    """Return list of (image_path, mask_path, stem) from dataset config."""
    img_dir = data_root / cfg["images"]
    mask_dir = data_root / cfg["masks"]
    pattern = cfg["mask_pattern"]

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


def build_model(backbone, device):
    model = torch.hub.load("facebookresearch/dinov2", backbone, verbose=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return model


def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def compute_patch_labels(mask_path, size, patch_size):
    """Resize mask to size x size, compute per-patch foreground fraction."""
    mask = Image.open(mask_path).convert("L")
    orig_w, orig_h = mask.size
    mask = mask.resize((size, size), Image.NEAREST)
    m = np.array(mask, dtype=np.float32)
    m = (m >= 128).astype(np.float32)

    grid = size // patch_size
    fractions = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = m[i * patch_size:(i + 1) * patch_size,
                      j * patch_size:(j + 1) * patch_size]
            fractions[i, j] = patch.mean()

    return fractions.reshape(-1), orig_w, orig_h


def extract_dataset(model, device, transform, pairs, size, patch_size, out_path, cost_writer, dataset_name):
    """Extract features for one dataset and save as .npz."""
    n_patches = (size // patch_size) ** 2
    feat_dim = 384 if "vits14" in out_path.parent.name else 768

    all_cls = []
    all_patches = []
    all_frac = []
    all_binary = []
    filenames = []
    mask_paths = []
    orig_sizes = []

    torch.cuda.reset_peak_memory_stats(device)

    for img_path, mask_path, stem in tqdm(pairs, desc=dataset_name):
        t0 = time.time()

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        if device.type == "cuda":
            x = x.half()

        with torch.no_grad():
            out = model.forward_features(x)

        cls_tok = out["x_norm_clstoken"][0].float().cpu().numpy()
        patch_tok = out["x_norm_patchtokens"][0].float().cpu().numpy()

        frac, orig_w, orig_h = compute_patch_labels(mask_path, size, patch_size)
        binary = (frac > 0.5).astype(np.uint8)

        all_cls.append(cls_tok.astype(np.float16))
        all_patches.append(patch_tok.astype(np.float16))
        all_frac.append(frac.astype(np.float16))
        all_binary.append(binary)
        filenames.append(stem)
        mask_paths.append(str(mask_path))
        orig_sizes.append((orig_w, orig_h))

        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        cost_writer.writerow([dataset_name, stem, f"{elapsed:.3f}", f"{peak_mb:.0f}"])

    cls_arr = np.stack(all_cls)
    patches_arr = np.stack(all_patches)
    frac_arr = np.stack(all_frac)
    binary_arr = np.stack(all_binary)
    orig_sizes_arr = np.array(orig_sizes, dtype=np.int32)

    # Write to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(dir=out_path.parent, suffix=".npz")
    os.close(fd)
    np.savez_compressed(
        tmp_path,
        cls_tokens=cls_arr,
        patch_tokens=patches_arr,
        patch_fg_fraction=frac_arr,
        patch_fg_binary=binary_arr,
        filenames=np.array(filenames),
        mask_paths=np.array(mask_paths),
        orig_sizes=orig_sizes_arr,
    )
    os.replace(tmp_path, out_path)

    return cls_arr.shape, patches_arr.shape, frac_arr.shape, binary_arr.shape


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument("--backbone", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Dataset keys from configs/datasets.yaml (default: all)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (currently processes one at a time for simplicity)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    args = parser.parse_args()

    patch_size = int(args.backbone.split("_vit")[1].replace("s", "").replace("b", ""))
    config = load_config(args.config)
    data_root = ROOT / config["data_root"]

    out_dir = ROOT / "features" / f"{args.backbone}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_dir = ROOT / "outputs" / "02_baselines"
    cost_dir.mkdir(parents=True, exist_ok=True)
    cost_path = cost_dir / "extraction_cost.csv"
    cost_exists = cost_path.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}, size: {args.size}, patch_size: {patch_size}")

    model = build_model(args.backbone, device)

    datasets_to_run = args.dataset or list(config["datasets"].keys())

    cost_file = open(cost_path, "a", newline="")
    cost_writer = csv.writer(cost_file)
    if not cost_exists:
        cost_writer.writerow(["dataset", "image", "time_s", "peak_gpu_mb"])

    for ds_name in datasets_to_run:
        out_path = out_dir / f"{ds_name}.npz"
        if out_path.exists():
            print(f"\nSkipping {ds_name} — {out_path} already exists")
            continue

        ds_cfg = config["datasets"][ds_name]
        pairs = get_image_mask_pairs(ds_cfg, data_root)
        print(f"\n{ds_name}: {len(pairs)} image-mask pairs")

        cls_shape, patch_shape, frac_shape, binary_shape = extract_dataset(
            model, device, build_transform(args.size),
            pairs, args.size, patch_size, out_path,
            cost_writer, ds_name,
        )

        file_size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path}")
        print(f"  CLS tokens:     {cls_shape}")
        print(f"  Patch tokens:   {patch_shape}")
        print(f"  Patch fg frac:  {frac_shape}")
        print(f"  Patch fg binary:{binary_shape}")
        print(f"  File size:      {file_size_mb:.1f} MB")

        # Sanity check: patch labels not all zero
        data = np.load(out_path)
        fg_sum = data["patch_fg_binary"].sum()
        n_positive_images = (data["patch_fg_binary"].sum(axis=1) > 0).sum()
        print(f"  Sanity: {fg_sum} foreground patches across {n_positive_images}/{len(pairs)} images")
        if fg_sum == 0:
            print("  WARNING: all patch labels are zero!")

    cost_file.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
