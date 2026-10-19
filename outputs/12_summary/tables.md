## Table 1: Resolution Comparison (224 vs 448)

Positives-only Dice, mean over 5 seeds (seeds 42-46). Features: DINOv2 ViT-S/14.

| Site | Dice 224 | Dice 448 | Delta |
|---|---|---|---|
| kvasir_val | 0.839 | 0.871 | +0.031 |
| polypgen_C1 | 0.715 | 0.735 | +0.020 |
| polypgen_C2 | 0.706 | 0.736 | +0.030 |
| polypgen_C3 | 0.765 | 0.808 | +0.043 |
| polypgen_C4 | 0.435 | 0.478 | +0.043 |
| polypgen_C5 | 0.464 | 0.572 | +0.108 |
| polypgen_C6 | 0.696 | 0.756 | +0.059 |

## Table 2: Ablation — Positives-Only Dice (5 seeds, original mask resolution)

Mean +/- std over seeds 42-46. Evaluation at original GT mask resolution via bilinear upsample.

| Site | Source 448 | Ensemble | Uniform PL | Agree 224-448 | Agree 448-448 |
|---|---|---|---|---|---|
| C1 | 0.733 +/- 0.005 | 0.746 +/- 0.003 | 0.744 +/- 0.013 | 0.754 +/- 0.011 | 0.746 +/- 0.007 |
| C2 | 0.734 +/- 0.013 | 0.749 +/- 0.005 | 0.748 +/- 0.009 | 0.761 +/- 0.011 | 0.754 +/- 0.010 |
| C3 | 0.807 +/- 0.003 | 0.809 +/- 0.002 | 0.809 +/- 0.005 | 0.808 +/- 0.008 | 0.811 +/- 0.005 |
| C4 | 0.479 +/- 0.008 | 0.478 +/- 0.006 | 0.491 +/- 0.009 | 0.496 +/- 0.007 | 0.502 +/- 0.006 |
| C5 | 0.570 +/- 0.007 | 0.540 +/- 0.010 | 0.579 +/- 0.006 | 0.567 +/- 0.015 | 0.592 +/- 0.011 |
| C6 | 0.760 +/- 0.013 | 0.759 +/- 0.008 | 0.762 +/- 0.015 | 0.775 +/- 0.016 | 0.765 +/- 0.011 |
| **Mean** | 0.681 | 0.680 | 0.689 | 0.694 | 0.695 |

## Table 3: Ablation — Delta vs Source 448

Win = positive mean delta. Bold = best variant per site.

| Site | Ensemble | Uniform PL | Agree 224-448 | Agree 448-448 |
|---|---|---|---|---|
| C1 | +0.0128 | +0.0110 | **+0.0214** | +0.0135 |
| C2 | +0.0147 | +0.0142 | **+0.0269** | +0.0194 |
| C3 | +0.0022 | +0.0020 | +0.0017 | **+0.0048** |
| C4 | -0.0008 | +0.0122 | +0.0171 | **+0.0229** |
| C5 | -0.0294 | +0.0089 | -0.0026 | **+0.0224** |
| C6 | -0.0015 | +0.0018 | **+0.0144** | +0.0045 |
| **Mean** | -0.0004 | +0.0083 | +0.0132 | +0.0146 |
| **Wins** | 3/6 | 6/6 | 5/6 | 6/6 |

## Table 4: Delta vs Uniform Self-Training

Shows marginal gain of agreement weighting over uniform pseudo-labels.

| Site | Uniform PL Dice | Agree 224-448 delta | Agree 448-448 delta |
|---|---|---|---|
| C1 | 0.7440 | +0.0104 | +0.0025 |
| C2 | 0.7483 | +0.0128 | +0.0053 |
| C3 | 0.8087 | -0.0003 | +0.0028 |
| C4 | 0.4911 | +0.0049 | +0.0107 |
| C5 | 0.5788 | -0.0115 | +0.0134 |
| C6 | 0.7621 | +0.0126 | +0.0028 |

## Table 5: False-Positive Rates on Empty-Mask Images

FP rate = fraction of empty-mask images where model predicts any foreground.
n_neg = number of images with empty GT mask at that site.

| Site | n_neg | Source 448 FP% | Ensemble FP% | Uniform PL FP% | Agree 224-448 FP% | Agree 448-448 FP% |
|---|---|---|---|---|---|---|
| C1 | 5 | 56% | 48% | 48% | 44% | 52% |
| C2 | 31 | 77% | 59% | 71% | 65% | 70% |
| C3 | 1 | 0% | 0% | 0% | 0% | 0% |
| C4 | 81 | 67% | 49% | 65% | 53% | 63% |
| C5 | 2 | 60% | 80% | 60% | 60% | 50% |
| C6 | 5 | 40% | 16% | 32% | 28% | 32% |

## Table 6: All-Images Dice (mean over 5 seeds)

Includes empty-mask images (Dice=1 if both pred and GT are empty, else 0).

| Site | Source 448 | Ensemble | Uniform PL | Agree 224-448 | Agree 448-448 |
|---|---|---|---|---|---|
| C1 | 0.727 | 0.741 | 0.740 | 0.751 | 0.741 |
| C2 | 0.682 | 0.714 | 0.701 | 0.719 | 0.707 |
| C3 | 0.807 | 0.809 | 0.809 | 0.809 | 0.812 |
| C4 | 0.424 | 0.490 | 0.440 | 0.486 | 0.456 |
| C5 | 0.568 | 0.537 | 0.577 | 0.566 | 0.591 |
| C6 | 0.751 | 0.763 | 0.757 | 0.772 | 0.760 |

## Table 7: Adaptation Cost

Adapt time = mean seconds per site for the self-training step (over 5 seeds).

| Variant | Adapt time (s) | Extra params |
|---|---|---|
| Source 448 | 0.0 | 0 |
| Ensemble | 0.0 | 0 |
| Uniform PL | 1.0 | 535,233 |
| Agree 224-448 | 1.1 | 535,233 |
| Agree 448-448 | 1.1 | 535,233 |

## Table 8: Few-Label Supervised Fine-Tuning (224 features)

Positives-only Dice, mean over 5 seeds. Decoder fine-tuned from Kvasir source weights.
Features: DINOv2 ViT-S/14 at 224x224 (16x16 patch grid).

| Site | Source-only | k=5 | k=10 | k=20 | k=50 |
|---|---|---|---|---|---|
| C1 | 0.722 | 0.688 | 0.705 | 0.714 | 0.727 |
| C2 | 0.705 | 0.710 | 0.716 | 0.705 | 0.735 |
| C3 | 0.773 | 0.741 | 0.749 | 0.756 | 0.768 |
| C4 | 0.445 | 0.396 | 0.428 | 0.445 | 0.466 |
| C5 | 0.450 | 0.443 | 0.480 | 0.490 | 0.509 |
| C6 | 0.685 | 0.676 | 0.692 | 0.691 | 0.717 |

## Note: PoC vs Ablation Discrepancy

The PoC script (poc_multires_5seeds.py) evaluates at decoder output resolution
(128x128 for 224 features, 256x256 for 448 features). The ablation script
(ablation_multires.py) evaluates at original GT mask resolution using bilinear
upsampling of probabilities. The ablation numbers are the official results.
Additionally, the PoC uses nearest-neighbor interpolation on binary masks for
agreement computation, while the ablation uses bilinear interpolation on
probabilities before thresholding.

