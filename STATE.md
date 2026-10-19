# STATE.md — Project Status (updated 2026-10-18)

## Pipeline
- Frozen DINOv2 (dinov2_vits14, 224x224, 16x16 patches, 384-d) -> cached features in `features/dinov2_vits14_224/`
- Two heads: (a) logistic regression patch probe, (b) 3-block conv decoder (16->128, ~535K params, has BatchNorm2d)
- Train on Kvasir-SEG (800 train / 200 val), test on PolypGen C1-C6

## Reproduced anchors (Phase 0)
- Quantization ceiling (Kvasir val): 0.9156 (expected ~0.916) OK
- Decoder Kvasir-val Dice: 0.8402 (expected ~0.840) OK
- Decoder C4 Dice: 0.4424 (from checkpoint; audit script trains fresh -> ~0.416)
- All tests pass (16/16)

## Known bugs (all fixed, tests exist)
1. float64->PIL mode 'F' garbage: fixed in `eval.py:upsample_probs` (casts to float32)
2. Wrong reshape: `reshape(384,16,16)` -> correct: `reshape(16,16,384).transpose(2,0,1)`
3. Decoder evaluation float64 bug: same fix as #1
4. Saerens EM source prior: must use actual training prior, not probs.mean()
5. C4 FOV teal panel: denominator inflated in FOV heuristic
6. C3 filename typo: one image skipped (456 of 457)
7. Kvasir has no negative frames

## Prior correction results (from audit_experiment.py)
- LR (balanced, pi_source=0.5): Kvasir-val 0.765, oracle/EM help C2 (+0.07) and C4 (+0.19), hurt C1/C3/C5/C6
- Decoder: Kvasir-val 0.840, oracle changes Dice by only -0.004 to +0.01 per site
- Prior shift is NOT the dominant error; appearance shift is (0.17-0.41 Dice drop)

## CORAL results (from coral_poc.py)
- CORAL alone generally hurts (decoder C1: -0.026, C4: -0.134)
- CORAL+EM sometimes helps but inconsistently

## Current work — ALL PHASES COMPLETE
- Phase 0: DONE — anchors reproduced, 19 tests pass
- Phase 1: DONE — prior correction headroom small for decoder (<0.01 Dice)
- Phase 2: DONE — channel_std, CORAL, AdaBN all NEGATIVE (0/6 wins each)
- Phase 3: DONE — prior-matched variants also NEGATIVE (0/6 wins each)
- Phase 4: DONE — RESULTS.md, 4 figures, reproduce.sh generated

## Headline result
**Negative result.** No label-free marginal alignment method improves cross-center
Dice on DINOv2 features. All three methods (channel standardization, CORAL, AdaBN)
hurt performance on most or all PolypGen centers. Prior-matched alignment reduces
the damage but never produces a net improvement. The cross-center gap (0.17–0.41
Dice) is not a marginal feature-distribution shift.

## Phase 2 summary (5 seeds)
| Variant | Mean delta across C1-C6 |
|---|---|
| B_channel_std | -0.018 |
| C_coral | -0.025 |
| D_adabn | -0.036 |

## Control experiments (outputs/08_controls/, 2026-10-18)

### Item 0: RESULTS.md fix
Prior headroom explanation corrected. Source pi=0.152, targets 0.055–0.105.
Logit shift equivalent to threshold change; oracle delta -0.005 to +0.010.

### Item 1: Resolution comparison (ctrl_resolution.csv)
128×128 vs original mask resolution: 0 differences > 0.01 in dice_pos.
128×128 evaluation is faithful for all prior results.

### Item 2: Positive control (ctrl_positive.csv)
Decomposed synthetic shift on Kvasir val:
| Shift type | Condition | channel_std | CORAL | AdaBN |
|---|---|---|---|---|
| scale+offset only | 1.0 | 101% | 99% | 91% |
| mild mixing | 3.2 | 92% | 86% | 89% |
| full mixing | 491.1 | 14% | 11% | 11% |
Oracle retrained: 100%. No code bug; methods fail on aggressive cross-channel mixing.

### Item 3: FOV-restricted statistics (ctrl_fov.csv)
FOV filtering makes negligible difference. C3/C4: 100% FOV.
Largest impact: C6 (66.8% FOV), FOV-restricted channel_std delta=-0.016 vs -0.033 all patches.
All alignment deltas remain negative.

### Item 4: Size analysis (ctrl_size_*.csv)
Cross-site gap persists within matched polyp-size bins.
Small polyps (<2×2 patches): Kvasir 2%, C1 12%, C4 25%, C5 32%.
Quantization ceiling drops for small polyps (<1% area: 0.32–0.65).

### Item 5: 448 vs 224 resolution (ctrl_448_comparison.csv)
| Site | 224 Dice | 448 Dice | Delta | Ceiling 224 | Ceiling 448 |
|---|---|---|---|---|---|
| kvasir_val | 0.839 | 0.871 | +0.031 | 0.902 | 0.951 |
| C1 | 0.715 | 0.735 | +0.020 | 0.850 | 0.926 |
| C2 | 0.706 | 0.736 | +0.030 | 0.865 | 0.932 |
| C3 | 0.765 | 0.808 | +0.043 | 0.844 | 0.922 |
| C4 | 0.435 | 0.478 | +0.043 | 0.773 | 0.893 |
| C5 | 0.464 | 0.572 | +0.108 | 0.760 | 0.887 |
| C6 | 0.696 | 0.756 | +0.059 | 0.793 | 0.906 |
Cost: 448 extraction 220s, training 859s (vs 224 cached/222s). Val dice 0.871 vs 0.840.

### Item 6: Few-label adaptation (ctrl_fewlabel.csv, ctrl_fewlabel_pseudo.csv)
Fine-tune decoder from source weights (50 steps) vs scratch, 5 seeds:
| Site | source_only | ft k=5 | ft k=10 | ft k=50 | scratch k=50 |
|---|---|---|---|---|---|
| C1 | 0.722 | 0.688 | 0.705 | 0.727 | 0.622 |
| C2 | 0.705 | 0.711 | 0.716 | 0.735 | 0.654 |
| C3 | 0.773 | 0.741 | 0.749 | 0.768 | 0.678 |
| C4 | 0.445 | 0.396 | 0.428 | 0.466 | 0.425 |
| C5 | 0.450 | 0.443 | 0.480 | 0.510 | 0.456 |
| C6 | 0.685 | 0.676 | 0.692 | 0.717 | 0.645 |
Fine-tuning with k≥10 matches source-only; k=50 improves on 4/6 sites.
Pseudo-label (k=0): comparable to source-only. Pseudo k=10: best on C5 (0.509) and C6 (0.732).

### Item 7: 07_reconcile addenda
NOT RUN: per-site ceilings, PolypDINO-style metrics, flip TTA, DyNA-protocol.
Listed in outputs/08_controls/pending_07_reconcile.txt.

### Item 8: Regression tests
24/24 tests pass. Added: self-alignment identity (channel_std, CORAL, AdaBN),
positive control recovery >50% (channel_std on scale+offset, CORAL on mild mixing).

## Key paths
- Features: `features/dinov2_vits14_224/{kvasir,polypgen_C1,...C6}.npz`
- Checkpoint: `checkpoints/decoder_corrected.pt`
- Metrics: `src/eval.py` (canonical, no duplicates)
- Decoder: `src/patch_decoder.py`
- Split: `data/splits/kvasir_train_val.json`
- Outputs: `outputs/06_poc/`
