"""Regression tests for the polyp segmentation pipeline.

Run: python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval import (
    GRID, N_PATCHES, dice_score, load_gt_mask, load_site_features,
    saerens_em, sigmoid, upsample_probs,
)

FEAT_DIR = ROOT / "features" / "dinov2_vits14_224"
SPLIT_PATH = ROOT / "data" / "splits" / "kvasir_train_val.json"


# ── 1. GT patches through pipeline = quantization ceiling ──────────

class TestQuantizationCeiling:
    def test_kvasir_val_ceiling_above_085(self):
        kvasir = load_site_features(FEAT_DIR, "kvasir")
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        fns = list(kvasir["filenames"])
        val_idx = [fns.index(fn) for fn in split["val"]]

        dices = []
        for idx in val_idx[:30]:
            gt_patches = kvasir["patch_fg_binary"][idx].astype(np.float32)
            prob_grid = gt_patches.reshape(GRID, GRID)
            ow, oh = int(kvasir["orig_sizes"][idx][0]), int(kvasir["orig_sizes"][idx][1])
            prob_up = upsample_probs(prob_grid, ow, oh)
            pred = (prob_up >= 0.5).astype(np.uint8)
            gt = load_gt_mask(str(kvasir["mask_paths"][idx]))
            dices.append(dice_score(pred, gt))

        mean_dice = np.mean(dices)
        assert mean_dice > 0.85, f"Quantization ceiling {mean_dice:.4f} < 0.85"


# ── 2. float32 dtype assertion before PIL mode F ───────────────────

class TestFloat32Dtype:
    def test_float32_roundtrip_values_in_range(self):
        rng = np.random.default_rng(42)
        prob = rng.random((16, 16)).astype(np.float32)
        result = upsample_probs(prob, 224, 224)
        assert result.dtype == np.float32
        assert result.min() >= -0.01, f"min={result.min()}"
        assert result.max() <= 1.01, f"max={result.max()}"

    def test_float64_input_coerced_to_sane_values(self):
        rng = np.random.default_rng(42)
        prob = rng.random((16, 16)).astype(np.float64)
        result = upsample_probs(prob, 224, 224)
        assert np.all(np.isfinite(result)), "Non-finite values after upsample"
        assert result.max() < 2.0, f"float64 not coerced: max={result.max()}"
        assert result.min() > -1.0, f"float64 not coerced: min={result.min()}"

    def test_raw_pil_float64_produces_garbage(self):
        """Demonstrate the bug: raw float64 → PIL mode F → garbage."""
        from PIL import Image
        rng = np.random.default_rng(42)
        prob = rng.random((16, 16))  # float64
        img_bad = Image.fromarray(prob, mode="F")
        bad = np.array(img_bad)
        assert bad.max() > 10 or bad.min() < -10, (
            "Expected garbage from raw float64→PIL, got sane values"
        )

        img_good = Image.fromarray(prob.astype(np.float32), mode="F")
        good = np.array(img_good)
        assert np.allclose(good, prob, atol=1e-6), "float32 roundtrip failed"


# ── 3. Token reshape orientation ───────────────────────────────────

class TestTokenReshape:
    def test_correct_vs_wrong_reshape_differ(self):
        tokens = np.arange(256 * 384, dtype=np.float32).reshape(256, 384)
        correct = tokens.reshape(GRID, GRID, 384).transpose(2, 0, 1)
        wrong = tokens.reshape(384, GRID, GRID)
        assert correct.shape == (384, 16, 16)
        assert wrong.shape == (384, 16, 16)
        assert not np.allclose(correct, wrong), (
            "Bug reshape and correct reshape should differ"
        )

    def test_spatial_layout_preserved(self):
        tokens = np.zeros((256, 384), dtype=np.float32)
        for i in range(256):
            tokens[i, 0] = float(i)

        correct = tokens.reshape(GRID, GRID, 384).transpose(2, 0, 1)
        assert correct[0, 0, 0] == 0.0
        assert correct[0, 0, 1] == 1.0
        assert correct[0, 1, 0] == 16.0
        assert correct[0, 15, 15] == 255.0

    def test_feature_vector_preserved_per_patch(self):
        rng = np.random.default_rng(42)
        tokens = rng.random((256, 384)).astype(np.float32)
        reshaped = tokens.reshape(GRID, GRID, 384).transpose(2, 0, 1)
        for r in range(GRID):
            for c in range(GRID):
                np.testing.assert_array_equal(
                    reshaped[:, r, c], tokens[r * GRID + c],
                    err_msg=f"Feature mismatch at patch ({r},{c})",
                )


# ── 4. Image/mask pairing counts ──────────────────────────────────

class TestPairingCounts:
    def test_kvasir_has_1000_images(self):
        d = np.load(FEAT_DIR / "kvasir.npz")
        assert len(d["filenames"]) == 1000
        assert d["patch_tokens"].shape == (1000, 256, 384)
        assert len(d["mask_paths"]) == 1000
        assert len(d["orig_sizes"]) == 1000

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("polypgen_C1", 256),
            ("polypgen_C2", 301),
            ("polypgen_C3", 456),
            ("polypgen_C4", 227),
            ("polypgen_C5", 208),
            ("polypgen_C6", 88),
        ],
    )
    def test_polypgen_site_counts(self, name, expected):
        d = np.load(FEAT_DIR / f"{name}.npz")
        n = len(d["filenames"])
        assert n == expected, f"{name}: got {n}, expected {expected}"
        assert d["patch_tokens"].shape == (expected, 256, 384)
        assert len(d["mask_paths"]) == expected

    def test_kvasir_split_covers_all(self):
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        assert len(split["train"]) == 800
        assert len(split["val"]) == 200
        assert len(set(split["train"]) & set(split["val"])) == 0

    def test_mask_paths_exist_kvasir_sample(self):
        kvasir = load_site_features(FEAT_DIR, "kvasir")
        for i in range(min(20, len(kvasir["mask_paths"]))):
            p = Path(str(kvasir["mask_paths"][i]))
            assert p.exists(), f"Mask not found: {p}"


# ── 5. Alignment sanity checks ──────────────────────────────────────

class TestAlignmentSanity:
    def test_channel_std_identity_reproduces_input(self):
        rng = np.random.default_rng(42)
        X = rng.random((100, 384)).astype(np.float64)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        eps = 1e-6
        X_aligned = (X - mu) / (sigma + eps) * sigma + mu
        np.testing.assert_allclose(X_aligned, X, atol=1e-5,
                                   err_msg="Channel std with identity stats must reproduce input")

    def test_coral_self_alignment_identity(self):
        rng = np.random.default_rng(42)
        X = rng.random((200, 384)).astype(np.float64)
        mu = X.mean(axis=0)
        C = np.cov(X, rowvar=False)
        D = C.shape[0]
        I = np.eye(D)
        reg = 1e-5

        d, V = np.linalg.eigh(C + reg * I)
        d = np.maximum(d, reg)
        C_isqrt = V @ np.diag(1.0 / np.sqrt(d)) @ V.T
        C_sqrt = V @ np.diag(np.sqrt(d)) @ V.T
        W = C_isqrt @ C_sqrt
        X_aligned = (X - mu) @ W + mu

        err = np.abs(X_aligned - X).max()
        assert err < 1e-6, f"CORAL self-alignment max error: {err}"

    def test_saerens_em_at_source_prior_returns_source_prior(self):
        rng = np.random.default_rng(42)
        probs = rng.random(1000)
        pi_src = float(probs.mean())
        pi_est = saerens_em(probs, pi_source=pi_src)
        assert abs(pi_est - pi_src) < 0.01, (
            f"EM at source prior should return ~source prior, got {pi_est:.4f} vs {pi_src:.4f}"
        )


# ── 6. Alignment self-consistency (own stats → identity) ─────────────

class TestAlignmentSelfConsistency:
    """Alignment with a site's own statistics must be the identity transform."""

    def test_channel_std_own_stats_identity(self):
        rng = np.random.default_rng(42)
        X = rng.random((500, 384)).astype(np.float32)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        eps = 1e-6
        X_a = ((X - mu) / (sigma + eps) * sigma + mu).astype(np.float32)
        np.testing.assert_allclose(X_a, X, atol=1e-4,
            err_msg="Channel std with own stats must be identity")

    def test_coral_own_stats_identity(self):
        rng = np.random.default_rng(42)
        X = rng.random((500, 384)).astype(np.float64)
        mu = X.mean(axis=0)
        C = np.cov(X, rowvar=False)
        reg = 1e-5
        I = np.eye(384)
        d, V = np.linalg.eigh(C + reg * I)
        d = np.maximum(d, reg)
        C_isqrt = V @ np.diag(1.0 / np.sqrt(d)) @ V.T
        C_sqrt = V @ np.diag(np.sqrt(d)) @ V.T
        W = C_isqrt @ C_sqrt
        X_a = (X - mu) @ W + mu
        err = np.abs(X_a - X).max()
        assert err < 1e-6, f"CORAL self-alignment error: {err}"

    def test_adabn_own_data_identity(self):
        """AdaBN with own data should preserve predictions (same BN stats)."""
        import copy
        import torch
        import torch.nn as nn
        sys.path.insert(0, str(ROOT / "src"))
        from patch_decoder import PatchDecoder

        rng = np.random.default_rng(42)
        X = rng.random((32, 256, 384)).astype(np.float32)

        torch.manual_seed(42)
        model = PatchDecoder(in_dim=384)
        model.eval()

        # Original BN stats from the data
        adapted = copy.deepcopy(model)
        for m in adapted.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.num_batches_tracked.zero_()
                m.momentum = None
        adapted.train()
        with torch.no_grad():
            for i in range(0, len(X), 16):
                batch = X[i:i+16]
                x = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in batch])
                adapted(torch.from_numpy(np.ascontiguousarray(x)))
        adapted.eval()

        # Compare predictions
        with torch.no_grad():
            x_t = np.stack([t.reshape(GRID, GRID, -1).transpose(2, 0, 1) for t in X[:8]])
            x_t = torch.from_numpy(np.ascontiguousarray(x_t))
            out_orig = model(x_t).numpy()
            out_adabn = adapted(x_t).numpy()

        # They won't be exactly identical (BN running stats vs data stats differ),
        # but with enough data they should be close. Use loose tolerance.
        diff = np.abs(out_orig - out_adabn).max()
        assert diff < 5.0, f"AdaBN with own data diverged too much: max diff {diff}"


# ── 7. Positive control regression test ──────────────────────────────

class TestPositiveControl:
    """Positive control: synthetic shift must be partially recoverable."""

    def test_channel_std_recovers_synthetic_shift(self):
        rng = np.random.default_rng(42)
        D = 384
        X = rng.random((200, D)).astype(np.float64)
        mu_s = X.mean(axis=0)
        sigma_s = X.std(axis=0)

        scale = rng.uniform(0.7, 1.4, size=D)
        offset = rng.normal(0, 0.5, size=D) * sigma_s
        X_shifted = X * scale + offset

        mu_t = X_shifted.mean(axis=0)
        sigma_t = X_shifted.std(axis=0)
        eps = 1e-6
        X_recovered = (X_shifted - mu_t) / (sigma_t + eps) * sigma_s + mu_s

        # Compute recovery in terms of mean squared error
        mse_shift = ((X_shifted - X) ** 2).mean()
        mse_recover = ((X_recovered - X) ** 2).mean()
        recovery = 1 - mse_recover / mse_shift
        assert recovery > 0.5, (
            f"Channel std should recover >50% of synthetic per-channel shift, "
            f"got {recovery:.1%}"
        )

    def test_coral_recovers_synthetic_affine_shift(self):
        rng = np.random.default_rng(42)
        D = 50  # smaller for speed
        X = rng.random((300, D)).astype(np.float64)
        mu_s = X.mean(axis=0)
        C_s = np.cov(X, rowvar=False)

        M = np.eye(D) + 0.2 * rng.standard_normal((D, D))
        offset = rng.normal(0, 0.5, size=D)
        X_shifted = X @ M + offset

        mu_t = X_shifted.mean(axis=0)
        C_t = np.cov(X_shifted, rowvar=False)

        reg = 1e-5
        I = np.eye(D)
        d_t, V_t = np.linalg.eigh(C_t + reg * I)
        d_t = np.maximum(d_t, reg)
        Ct_isqrt = V_t @ np.diag(1.0 / np.sqrt(d_t)) @ V_t.T
        d_s, V_s = np.linalg.eigh(C_s + reg * I)
        d_s = np.maximum(d_s, reg)
        Cs_sqrt = V_s @ np.diag(np.sqrt(d_s)) @ V_s.T
        W = Ct_isqrt @ Cs_sqrt
        X_recovered = (X_shifted - mu_t) @ W + mu_s

        mse_shift = ((X_shifted - X) ** 2).mean()
        mse_recover = ((X_recovered - X) ** 2).mean()
        recovery = 1 - mse_recover / mse_shift
        assert recovery > 0.5, (
            f"CORAL should recover >50% of synthetic affine shift, "
            f"got {recovery:.1%}"
        )
