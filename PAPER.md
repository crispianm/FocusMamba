# FocusMamba Paper Notes (Living Document)

## 1) Project Summary

**Working title:** FocusMamba: ROI-Conditioned Spatiotemporal State Space Modeling for Video Focus Map Prediction

**Task:** Predict soft per-pixel focus maps over video clips, conditioned on scene content, depth-derived optical behavior, and an ROI prior.

**Core idea:** Combine physically-motivated synthetic supervision (Circle of Confusion from depth) with a Mamba-based encoder-decoder that alternates spatial and temporal sequence modeling, plus explicit ROI conditioning.

---

## 2) Problem Framing

### 2.1 Motivation
Conventional binary saliency supervision is a poor fit for depth-of-field behavior, which is inherently continuous and asymmetric around the focal plane. We model focus as a **soft field** induced by lens physics and evaluate with metrics that preserve this continuity.

### 2.2 Input / Output
- **Input:** video clip `frames` with shape `(B, C, T, H, W)`, ROI box `roi` with shape `(B, 4)` as `[x1,y1,x2,y2]` normalized to `[0,1]`.
- **Output:** soft focus map `focus_map` with shape `(B, 1, T, H, W)` in `[0,1]`.

---

## 3) Method Overview

### 3.1 Data supervision from depth and lens model
Per-pixel CoC is computed from depth:

\[
\mathrm{CoC}(d) = \frac{f^2}{N\,S_{focus}}\cdot\frac{|d-S_{focus}|}{d}
\]

where:
- \(f\): focal length sampled uniformly in `[24,85]` mm,
- \(N\): aperture sampled log-uniformly in `[1.4,8.0]`,
- \(S_{focus}\): sampled focal distance (Beta(2,2)-mapped over depth range),
- \(d\): per-pixel depth.

Focus target is:

\[
\mathrm{focus\_map} = 1 - \mathrm{clip}(\mathrm{CoC}/\mathrm{max\_coc}, 0, 1)
\]

### 3.2 Dynamic ROI generation (current implementation)
For each clip (after focus map generation):
1. Use middle frame `t = T//2`.
2. Find “perfectly in-focus” mask: `focus_map[t] > 0.95`.
3. Sample one pixel from that mask as simulated user tap.
4. Build random box around that point with width/height in 10%–30% of image size.
5. Clamp to image boundaries, normalize to `[0,1]`.
6. Fallback: if mask empty, use argmax focus pixel.

This creates ROI labels that are physically aligned with sampled focal plane and still stochastic enough to prevent overfitting to fixed box geometry.

### 3.3 Model architecture
- Tubelet embedding via 3D conv (`t_patch`, `patch_size`, `patch_size`).
- Sinusoidal 3D positional encoding.
- Additive ROI conditioning injected before first Mamba stage.
- Encoder alternates:
  - **Spatial Cross-Scan Mamba** (4 directions: LR, RL, TB, BT),
  - **Temporal Mamba** (1D along time).
- Multi-scale encoder-decoder with skip connections.
- Decoder includes spatial Cross-Scan Mamba and local depthwise-separable 3D smoothing.
- Temporal smoothing conv (`k=3`) before final sigmoid head.

---

## 4) Losses and Metrics

### 4.1 Training loss (weighted sum)
- L1 reconstruction,
- SSIM loss (`1 - SSIM`),
- Edge-aware gradient loss (Sobel-weighted),
- Temporal Gradient Matching (difference-of-differences across time).

Current default weights:
- `lambda_l1 = 1.0`
- `lambda_ssim = 0.5`
- `lambda_grad = 0.5`
- `lambda_tgm = 0.2`

### 4.2 Validation metrics
Implemented:
- MAE,
- SSIM,
- PSNR,
- weighted F-measure (threshold sweep, beta=0.3),
- Temporal Stability Score (TSS),
- optional LPIPS.

Rationale: avoid hard-threshold-only evaluation for naturally soft focus maps.

---

## 5) Training Protocol

- Mixed precision: bf16 autocast + GradScaler.
- Gradient clipping: max norm 1.0.
- LR schedule: linear warmup then cosine decay.
- Scene-level deterministic split (80/10/10).
- TensorBoard logging of losses and metrics.
- Best checkpoint selected by **validation MAE**.
- Baseline mode: per-frame 2D U-Net (`--baseline`) for ablation.

---

## 6) Current Code State (as of this snapshot)

### 6.1 Implemented components
- Data loader + CoC supervision + dynamic ROI generation.
- CoC distribution inspection utility.
- Mamba blocks (with fallback if `mamba_ssm` import unavailable).
- ROI conditioning module.
- Encoder + decoder + full FocusMamba model wrapper.
- Composite loss and metric suite.
- End-to-end training script with baseline mode.
- Config file with core hyperparameters.

### 6.2 Not fully paper-complete yet
- Large-scale experiment table and final benchmark numbers are not yet documented here.
- Final dependency/toolchain reproducibility for CUDA + Mamba custom ops may still depend on machine setup.
- Additional ablations (e.g., ROI on/off, cross-scan on/off, loss component ablations) should be run and tabulated.

---

## 7) Suggested Paper Contributions (wording draft)

1. **Physics-aware soft supervision** for focus prediction using depth-derived CoC and realistic lens parameter sampling.
2. **ROI-conditioned spatiotemporal SSM architecture** using additive spatial conditioning and cross-scan spatial Mamba.
3. **Temporal-consistency-aware objective** combining structural, boundary, and temporal gradient terms.
4. **Evaluation protocol aligned with soft focus maps** (threshold-sweep weighted F, TSS, SSIM, LPIPS).

---

## 8) Experimental Plan for Write-up

### 8.1 Main comparisons
- FocusMamba vs per-frame U-Net baseline.
- (Optional) FocusMamba vs temporal CNN/ConvLSTM variant.

### 8.2 Ablations to include
- Remove ROI conditioning.
- Replace dynamic ROI with static ROI.
- Remove cross-scan (single raster direction only).
- Remove TGM loss.
- Replace soft metrics with hard-threshold F1 to show mismatch.

### 8.3 Reporting
- Main metrics: MAE, SSIM, PSNR, weighted-F, TSS, LPIPS.
- Efficiency: parameter count and FLOPs estimate.
- Qualitative: temporal sequences showing boundary smoothness and reduced jitter.

---

## 9) Reproducibility Checklist

- Keep `config.yaml` frozen per experiment.
- Record commit hash for each run.
- Log train/val splits and random seed.
- Save `best.pt` and `latest.pt` with config snapshot.
- Export TensorBoard scalars to CSV for plotting.

---

## 10) Quick Links to Current Implementation

- `dataloader/focus_dataset.py`
- `dataloader/inspect_coc_distribution.py`
- `models/mamba_block.py`
- `models/roi_conditioning.py`
- `models/encoder.py`
- `models/decoder.py`
- `models/focus_mamba.py`
- `utils/loss.py`
- `utils/metrics.py`
- `train.py`
- `config.yaml`

---

## 11) Notes to Self Before Final Writing

- Replace all draft claims with measured numbers.
- Verify whether reported focus-map behavior generalizes across scenes and clip boundaries.
- Add failure cases (thin structures, sudden depth jumps, low-light frames).
- Include explicit discussion of synthetic-supervision bias due to depth and camera model assumptions.
