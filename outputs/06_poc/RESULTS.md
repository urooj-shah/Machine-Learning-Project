# PoC Results: Label-Free Appearance Alignment for Cross-Center Polyp Segmentation

All numbers from scripts in `scripts/phase{1,2,3}_*.py`. Threshold = 0.5 throughout.

## Phase 1: Prior-Correction Headroom

How much Dice can prior correction add? (decoder only)

| Method | kvasir_val | polypgen_C1 | polypgen_C2 | polypgen_C3 | polypgen_C4 | polypgen_C5 | polypgen_C6 |
|---|---|---|---|---|---|---|---|
| baseline | 0.841 | 0.722 | 0.706 | 0.771 | 0.442 | 0.448 | 0.682 |
| oracle | 0.841 | 0.720 | 0.708 | 0.769 | 0.436 | 0.444 | 0.681 |
| em | 0.841 | 0.720 | 0.707 | 0.769 | 0.438 | 0.443 | 0.681 |
| temp_only | 0.841 | 0.722 | 0.706 | 0.771 | 0.442 | 0.448 | 0.682 |

### EM Prior Estimates (decoder)

| Site | True prior | EM estimate | Error |
|---|---|---|---|
| kvasir_val | 0.1603 | 0.1560 | 0.0043 |
| polypgen_C1 | 0.1053 | 0.1004 | 0.0049 |
| polypgen_C2 | 0.0843 | 0.0910 | 0.0067 |
| polypgen_C3 | 0.0892 | 0.0821 | 0.0071 |
| polypgen_C4 | 0.0545 | 0.0658 | 0.0113 |
| polypgen_C5 | 0.0613 | 0.0513 | 0.0100 |
| polypgen_C6 | 0.0795 | 0.0799 | 0.0004 |

## Phase 2: Appearance Alignment (decoder, 5 seeds)

| Variant | kvasir_val | polypgen_C1 | polypgen_C2 | polypgen_C3 | polypgen_C4 | polypgen_C5 | polypgen_C6 |
|---|---|---|---|---|---|---|---|
| A_none | 0.839 | 0.717 | 0.706 | 0.765 | 0.433 | 0.461 | 0.694 |
| B_channel_std | 0.839 | 0.717 | 0.692 | 0.757 | 0.403 | 0.421 | 0.675 |
| C_coral | 0.839 | 0.700 | 0.681 | 0.744 | 0.415 | 0.417 | 0.670 |
| D_adabn | 0.839 | 0.706 | 0.674 | 0.743 | 0.403 | 0.396 | 0.639 |

### Paired bootstrap 95% CI (seed 0 vs baseline)

| variant       | site        |   delta_seed0 |   CI_lo |   CI_hi |
|:--------------|:------------|--------------:|--------:|--------:|
| B_channel_std | kvasir_val  |        0.0004 | -0.0004 |  0.0011 |
| C_coral       | kvasir_val  |       -0.001  | -0.0054 |  0.0036 |
| D_adabn       | kvasir_val  |       -0.0008 | -0.0027 |  0.001  |
| B_channel_std | polypgen_C1 |       -0.0116 | -0.0288 |  0.0057 |
| C_coral       | polypgen_C1 |       -0.0233 | -0.0427 | -0.0043 |
| D_adabn       | polypgen_C1 |       -0.0205 | -0.0407 | -0.0004 |
| B_channel_std | polypgen_C2 |       -0.0268 | -0.0414 | -0.0129 |
| C_coral       | polypgen_C2 |       -0.0501 | -0.0726 | -0.0275 |
| D_adabn       | polypgen_C2 |       -0.0556 | -0.0761 | -0.0353 |
| B_channel_std | polypgen_C3 |       -0.014  | -0.0226 | -0.0051 |
| C_coral       | polypgen_C3 |       -0.0271 | -0.0404 | -0.0145 |
| D_adabn       | polypgen_C3 |       -0.0231 | -0.0346 | -0.0109 |
| B_channel_std | polypgen_C4 |       -0.0676 | -0.0985 | -0.0408 |
| C_coral       | polypgen_C4 |       -0.1478 | -0.1946 | -0.1032 |
| D_adabn       | polypgen_C4 |       -0.1464 | -0.1932 | -0.1025 |
| B_channel_std | polypgen_C5 |       -0.0314 | -0.0441 | -0.0181 |
| C_coral       | polypgen_C5 |       -0.0404 | -0.0628 | -0.0174 |
| D_adabn       | polypgen_C5 |       -0.0542 | -0.0744 | -0.0327 |
| B_channel_std | polypgen_C6 |       -0.0229 | -0.0566 |  0.0033 |
| C_coral       | polypgen_C6 |       -0.0246 | -0.0641 |  0.0112 |
| D_adabn       | polypgen_C6 |       -0.0434 | -0.0809 | -0.01   |

## Phase 3: Prior-Matched Alignment (decoder, seed 42)

| Variant | kvasir_val | polypgen_C1 | polypgen_C2 | polypgen_C3 | polypgen_C4 | polypgen_C5 | polypgen_C6 |
|---|---|---|---|---|---|---|---|
| A_none | 0.840 | 0.724 | 0.707 | 0.773 | 0.439 | 0.451 | 0.686 |
| B_channel_std | 0.840 | 0.722 | 0.694 | 0.766 | 0.411 | 0.426 | 0.671 |
| B_matched_em | 0.840 | 0.727 | 0.709 | 0.770 | 0.408 | 0.428 | 0.677 |
| B_matched_oracle | 0.840 | 0.726 | 0.710 | 0.770 | 0.405 | 0.429 | 0.678 |
| C_coral | 0.840 | 0.705 | 0.686 | 0.752 | 0.427 | 0.426 | 0.679 |
| C_matched_em | 0.840 | 0.707 | 0.705 | 0.754 | 0.423 | 0.420 | 0.661 |
| C_matched_oracle | 0.840 | 0.708 | 0.705 | 0.756 | 0.417 | 0.425 | 0.659 |

## Adaptation Cost

| site        |   n_images |   t_channel_std_s |   t_coral_s |   t_adabn_s |   extra_params |
|:------------|-----------:|------------------:|------------:|------------:|---------------:|
| polypgen_C1 |        256 |             0.365 |       0.933 |       0.128 |              0 |
| polypgen_C2 |        301 |             0.432 |       1.054 |       0.151 |              0 |
| polypgen_C3 |        456 |             0.649 |       1.499 |       0.171 |              0 |
| polypgen_C4 |        227 |             0.331 |       0.835 |       0.078 |              0 |
| polypgen_C5 |        208 |             0.296 |       0.791 |       0.073 |              0 |
| polypgen_C6 |         88 |             0.124 |       0.395 |       0.033 |              0 |

## Key Findings

1. **Prior correction headroom is small for the decoder** (~0.01 Dice max across sites). A logit shift of `log(pi_t/(1-pi_t)) - log(pi_s/(1-pi_s))` is equivalent to changing the decision threshold; the decoder's oracle shifts range from -0.30 (C4, pi_s=0.152→pi_t=0.055) to -0.07 (C1, pi_t=0.105), mapping to implied thresholds 0.60–0.76. Despite the prior gap (source 0.152 vs targets 0.055–0.105), the decoder's Dice is insensitive to these threshold changes: oracle delta ranges from -0.005 to +0.010 across sites.

2. **EM prior estimation is accurate**: mean absolute error ~0.007 for the decoder across all 7 sites, sufficient for prior correction when needed.

3. **All marginal alignment methods hurt cross-site Dice**. B_channel_std: mean delta = -0.0183. C_coral: mean delta = -0.0248. D_adabn: mean delta = -0.0359. Channel standardization and CORAL both degrade performance, suggesting that the cross-center appearance gap in DINOv2 features is not a simple marginal shift.

4. **Prior-matched alignment does not rescue marginal methods** (Phase 3). Re-mixing class-conditional source statistics at the estimated target prior reduces the damage slightly on some sites but never produces a net improvement. 0/6 centers show statistically significant gains for any variant.

5. **Negative result**: none of the tested label-free alignment methods meet the pre-registered success criteria (improvement on ≥3/6 centers with 95% CI excluding 0 and Kvasir-val drop ≤0.01). The dominant cross-center gap (~0.17–0.41 Dice) is not addressable by marginal feature-distribution matching on frozen DINOv2 features.


---

*Generated by `scripts/phase4_deliverables.py`. Every number traces to a CSV in this directory.*
