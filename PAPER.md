# FocusMamba Paper Notes (Living Document)

## 1) Project Summary

**Working title:** FocusMamba: Degradation-Robust Metric Video Depth Estimation via Spatiotemporal State Space Modelling

**Task:** Predict dense, per-pixel metric depth (in metres) from video clips, with robustness to low-light and degraded imaging conditions, via knowledge distillation from an ensemble of frozen teacher depth models.

**Core idea:** A compact Mamba-based encoder-decoder student model learns temporally consistent metric depth by distilling knowledge from three complementary frozen teachers (Depth Anything V3, Apple Depth Pro, Video Depth Anything) that run on clean frames. The student alternates spatial cross-scan and causal temporal SSM blocks across multiple scales, enabling efficient spatiotemporal reasoning without self-attention. An optional degradation curriculum trains the student on synthetically corrupted input while teachers supervise from the clean version.

---

## 2) Problem Framing

### 2.1 Motivation
State-of-the-art monocular depth models achieve strong per-frame accuracy but often:
1. lack temporal consistency when applied to video,
2. degrade severely under adverse imaging conditions (low light, motion blur, sensor noise), and
3. are too large for real-time deployment.

We address all three by distilling an ensemble of large teacher models into a lightweight Mamba-based student that processes full video clips natively and can be trained on degraded inputs.

### 2.2 Input / Output
- **Input:** video clip `frames` with shape `(B, 3, T, H, W)`, float32 in `[0, 1]`.
- **Output:** `dict` containing:
  - `"depth"`: `(B, 1, T, H, W)` — metric depth in metres (positive, via `exp(log_depth)`).
  - `"uncertainty"` (optional): `(B, 1, T, H, W)` — per-pixel aleatoric log-variance.

---

## 3) Method Overview

### 3.1 Multi-Teacher Distillation

Three frozen teacher models provide pseudo ground-truth metric depth on clean (undegraded) frames at training time. The student model never sees hand-labelled depth — all supervision comes from teacher predictions.

| Teacher | Type | Resolution | Backbone | Key Capability |
|---------|------|-----------|----------|----------------|
| **Depth Anything V3** (da3metric-large) | Image | 518 × 518 | DinoV2 ViT-L + DPT | State-of-the-art metric depth |
| **Apple Depth Pro** | Image | 1536 × 1536 | ViT multi-scale | High-resolution canonical inverse depth + FoV → metric |
| **Video Depth Anything** (vitl) | Video | 518 × 518 | DinoV2 ViT-L + DPT Temporal | Native temporal modelling via motion modules |

All teacher model architectures are vendored locally under `models/teachers/vendor/` to eliminate external git repository dependencies. Each teacher wrapper (`models/teachers/`) normalises input, runs the frozen forward pass, and returns metric depth at the student's spatial resolution.

**Ensemble weighting:** In the current trial configuration, all teachers are equally weighted (L1 loss against each). The full distillation config (`configs/base.yaml`) supports confidence-weighted ensembling, where teacher agreement (measured via inter-teacher depth variance) modulates per-pixel loss strength.

### 3.2 Degradation Curriculum (planned, not yet active in trial)

A physically motivated `LowLightDegradation` pipeline (`dataloader/degradation.py`) simulates:
- **Signal-dependent Poisson noise** (shot noise, calibrated to lux level).
- **Gaussian read noise** (sensor read noise).
- **Motion blur** via random 2D trajectory kernel rasterisation.
- **Brightness reduction** to simulate low-light scenes.
- **Temporal variance** — per-frame jitter of noise parameters for realism.

A `CurriculumScheduler` (`training/curriculum.py`) ramps degradation severity from 0 → 1 over training (linear warmup, then cosine ramp). This forces the student to learn from easy (clean) to hard (heavily degraded) inputs while teachers always see the clean version.

### 3.3 Model Architecture

```
Video clip (B, 3, T, H, W)
        │
        ▼
 TubeletEmbedding           Conv3d(3, 96, k=(2,4,4), s=(2,4,4))
        │                   → (B, T/2, H/4, W/4, 96)
        ▼
 Sinusoidal 3-D PE          Additive, split C into three axes (T, H, W)
        │
        ▼
 FocusMambaEncoder ──────────────────────────────────────────
  Stage 0  (res/4,  C=96 )  [SpatialMamba + TemporalMamba] × 2  ┐
  Stage 1  (res/8,  C=192)  [SpatialMamba + TemporalMamba] × 2  │ skip
  Stage 2  (res/16, C=384)  [SpatialMamba + TemporalMamba] × 4  │ connections
  Stage 3  (res/32, C=768)  [SpatialMamba + TemporalMamba] × 2  ┘ (bottleneck)
        │
        ▼
 FocusMambaDecoder ──────────────────────────────────────────
  Stage 3: 768→384  (upsample 2× + skip from encoder stage 2)
  Stage 2: 384→192  (upsample 2× + skip from encoder stage 1)
  Stage 1: 192→96   (upsample 2× + skip from encoder stage 0)
  Stage 0: 96→96    (upsample 2×, no skip)
        │
        ▼
 Temporal Smoothing          Depthwise Conv3d(96, 96, k=(3,1,1), groups=96)
        │
        ▼
 Depth Head                  Conv3d(96, 1, k=1) → exp(·) → metric depth (m)
```

#### SpatialMambaBlock
Reshapes `(B, T, H, W, C)` → `(B×T, H×W, C)` and applies a `CrossScanMambaBlock`: four independent Mamba SSMs scan in four spatial directions (row LR, row RL, col TB, col BT). Each SSM's output is returned to row-major order, and the four are combined via learned softmax-normalised scan weights. Pre-RMSNorm + residual.

#### TemporalMambaBlock
Reshapes `(B, T, H, W, C)` → `(B×H×W, T, C)` and applies a single-direction causal Mamba SSM along the time axis. Pre-RMSNorm + residual.

#### Spatial Downsampling
`Conv3d(C, 2C, k=(1,2,2), s=(1,2,2))` — 2× spatial reduction, channels doubled, temporal dimension preserved.

#### Decoder Stage
Each stage: trilinear upsample 2× → 1×1 conv channel reduction → concatenate encoder skip → 1×1 conv project → `SpatialMambaBlock` → `DepthwiseSeparableConv3d` (3D DWConv + pointwise + GroupNorm + GELU).

#### Depth Head
The decoder produces log-depth, exponentiated (`exp(·)`) to enforce positive metric output. Optional uncertainty head predicts log-variance for aleatoric uncertainty modelling.

### 3.4 Ablation Baseline: FocusTransformer
A drop-in transformer variant (`models/focus_transformer.py`) replaces all Mamba SSMs with multi-head self-attention, using the same encoder-decoder structure. `utils/param_check.py` tunes the transformer's `mlp_ratio` so both models have matching parameter counts (within 5%).

---

## 4) Losses and Metrics

### 4.1 Training Loss

**Current production loss** (`training/losses/combined.py`, `configs/experiments/tartanair_v2.yaml`):

All six components are active simultaneously. GT SI-log and distillation are **additive** — when both ground-truth depth and cached teacher pseudo-labels are available (TartanAir v2 + teacher cache), both signals contribute to every training step:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{SI-log}}^{\text{GT}}}_{1.0} + \underbrace{\mathcal{L}_{\text{distill}}}_{1.0} + \underbrace{\mathcal{L}_{\text{grad}}}_{1.0} + \underbrace{\mathcal{L}_{\text{TGM}}}_{1.0} + \underbrace{0.2\,\mathcal{L}_{\text{NLL}}}_{0.2}$$

- **Scale-invariant log loss** (GT supervision and per-teacher distillation): $\text{SI-log} = \text{mean}(d^2) - \lambda \cdot \text{mean}(d)^2$, where $d = \log D_{\text{pred}} - \log D_{\text{gt}}$ and $\lambda = 0.5$. The squared (non-rooted) form is used for numerical stability — the $\sqrt{\cdot}$ variant has a $1/(2\sqrt{x})$ gradient that diverges as the loss converges.
- **Multi-teacher distillation** (`training/losses/distillation.py`): Cached pseudo-labels from DA3, Depth Pro, and VDA loaded by the DataLoader. Per-teacher SI-log losses are weight-averaged (weights 1.0 / 0.3 / 1.0). Confidence weighting via inter-teacher log-depth variance is computed but currently informational only.
- **Edge-aware gradient smoothness loss** (`training/losses/gradient.py`): L1 between spatial gradients of predicted and GT log-depth, evaluated at 4 scales. Encourages sharp depth boundaries aligned with the scene structure.
- **Temporal gradient matching (TGM) loss** (`training/losses/temporal.py`): From Video Depth Anything (Chen et al., 2025, Eq. 3). Penalises temporal depth oscillation in log-space, masked to static regions where $|\Delta_t \log D_{\text{gt}}| < 0.10$.
- **Uncertainty NLL**: Gaussian NLL with the model's predicted log-variance, encouraging calibrated aleatoric uncertainty. Weighted at 0.2 to prevent the uncertainty head from dominating early training.

Current weights (`configs/experiments/tartanair_v2.yaml`):
- `si_log_weight = 1.0`
- `distillation_weight = 1.0`
- `gradient_weight = 1.0`
- `temporal_weight = 1.0`
- `uncertainty_nll_weight = 0.2`

### 4.2 Evaluation Metrics

**Depth metrics** (`evaluation/metrics/depth_metrics.py`):
- AbsRel, SqRel, RMSE, RMSE_log, SI-log
- Threshold accuracy: δ < 1.25, δ < 1.25², δ < 1.25³

**Temporal metrics** (`evaluation/metrics/temporal_metrics.py`):
- Flicker score (mean absolute frame-to-frame depth difference)
- Optical-flow-weighted temporal consistency error (stub)

**Autofocus metrics** (`evaluation/metrics/autofocus_metrics.py`):
- Focus distance error, focus pull success rate, focus latency

---

## 5) Training Protocol

### 5.1 Current Working Pipeline (`test_training.py`)
Self-contained training script used for iterative development:

- **Data:** YouTube-VOS video clips (RGB only, no ground-truth depth). Teachers provide all supervision.
- **Loss:** Weighted L1 against each teacher's pseudo-GT depth.
- **Optimiser:** AdamW (lr=2e-4, weight_decay=0.01).
- **Scheduler:** Linear warmup (200 steps) → cosine decay.
- **Mixed precision:** bf16 autocast + GradScaler (on CUDA).
- **Gradient clipping:** max norm 1.0.
- **Logging:** TensorBoard — scalar losses + RGB/depth image grid visualisations every N steps, per-teacher mean depth monitoring.
- **Checkpointing:** Best (by validation L1) and latest saved every epoch. Full checkpoint includes model, optimiser, scaler, scheduler state + config snapshot.
- **Resume:** `--resume checkpoints/trial_youtube_vos/latest.pt`.

### 5.2 Full Training Script (`train.py`)
Primary production training entry point. Used for all TartanAir v2 and multi-teacher distillation runs.

- **Combined loss**: SI-log (GT) + multi-teacher distillation (cached pseudo-labels) + gradient smoothness + temporal consistency + uncertainty NLL, all active simultaneously.
- Distillation always enabled from cached teacher pseudo-labels — no live teacher inference required at training time.
- Degradation curriculum scheduling (inactive for GT-supervised runs).
- EMA of student weights (decay=0.999).
- NaN gradient guards on depth, total loss, and per-parameter gradients.
- `WarmupCosineScheduler` (warmup_steps=500 → cosine decay over 20 epochs).
- Latency profiling callback at end of training.
- Depth visualisation callbacks with turbo colourmap.

### 5.3 Hardware
Developed on **Isambard-AI** (NVIDIA GH200 Grace Hopper Superchip, ARM aarch64 + H200 GPU). SLURM job script: `jobs/run_test.sh`.

---

## 6) Data Pipeline

### 6.1 YouTubeVOSDataset (`dataloader/youtube_vos.py`)
Primary dataset. Loads RGB video clips from YouTube-VOS; no ground-truth depth (teachers provide pseudo-GT at runtime).
- Returns `"frames"` `(3, T, H, W)` float `[0, 1]`, video ID, start frame index.
- Configurable clip stride, frame stride, max videos, train/val split.

### 6.2 FocusDataset (`dataloader/focus_dataset.py`)
Legacy dataset for supervised training with depth maps + Circle-of-Confusion focus maps.
- Expects `{scene}/frames_sharp/` + `{scene}/depth_maps/depth_*.npz`.
- Synthesises focus maps on-the-fly via CoC lens model if precomputed maps absent.
- Returns `frames`, `focus_maps`, `depth_maps`, and scene metadata.

### 6.3 WildDataset (`dataloader/wild_dataset.py`)
Smartphone focal-sweep evaluation dataset. Loads from captured video + optional `labels.json` with per-frame focus distances.

### 6.4 LowLightDegradation (`dataloader/degradation.py`)
Fully implemented. Physically motivated low-light + motion blur simulation with per-clip temporal coherence and per-frame jitter.

---

## 7) Current Code State

### 7.1 Fully Implemented

| Component | Location | Notes |
|-----------|----------|-------|
| FocusMamba model | `models/focus_mamba.py` | Metric depth via exp(log_depth) |
| 4-stage encoder with cross-scan + temporal Mamba | `models/encoder.py` | 4-way spatial scan, causal temporal |
| UNet decoder with depth head | `models/decoder.py` | Temporal smoothing, optional uncertainty |
| Mamba blocks (spatial, temporal, cross-scan) | `models/mamba_block.py` | Graceful fallback if mamba_ssm absent |
| Transformer ablation baseline | `models/focus_transformer.py`, `models/transformer_block.py` | Drop-in replacement, param-matched |
| DA3 teacher (vendored) | `models/teachers/depth_anything_v3.py` + `vendor/depth_anything_v3/` | Hardcoded da3metric-large, 334.2M params |
| Depth Pro teacher (vendored) | `models/teachers/depth_pro.py` + `vendor/depth_pro/` | fp16, canonical inverse depth → metric |
| Video Depth Anything teacher (vendored) | `models/teachers/video_teacher.py` + `vendor/video_depth_anything/` | Native temporal processing, 384.4M params |
| Teacher registry + factory | `models/teachers/__init__.py` | `build_teacher(name, cfg)` |
| Trial training script | `test_training.py` | YouTube-VOS + L1 distillation, end-to-end |
| YouTube-VOS dataloader | `dataloader/youtube_vos.py` | RGB clips, no GT needed |
| Low-light degradation pipeline | `dataloader/degradation.py` | Poisson + Gaussian + motion blur |
| Degradation curriculum scheduler | `training/curriculum.py` | Linear warmup → cosine severity ramp |
| EMA model | `training/ema.py` | Exponential moving average (decay=0.999) |
| Scale-invariant log loss | `training/losses/scale_invariant.py` | SI-log squared form (no sqrt), λ=0.5 |
| Gradient smoothness loss | `training/losses/gradient.py` | Edge-aware L1 gradient matching, 4 scales, log-space |
| Temporal consistency loss | `training/losses/temporal.py` | TGM (Video Depth Anything Eq. 3), log-space, stability mask |
| Distillation loss | `training/losses/distillation.py` | Per-teacher SI-log + confidence weighting |
| Combined loss | `training/losses/combined.py` | GT + distillation additive; all 5 components active |
| Depth metrics | `evaluation/metrics/depth_metrics.py` | AbsRel, SqRel, RMSE, δ thresholds |
| Depth visualisation callback | `training/callbacks/visualise_depth.py` | Turbo colourmap for TensorBoard |
| Latency profiler | `training/callbacks/latency_profiler.py` | FPS measurement vs target |
| Realtime depth engine | `inference/realtime_engine.py` | Single-frame + clip processing, shot-cut detection |
| Autofocus interface | `inference/autofocus_interface.py` | Depth → focus distance (mm), temporal smoothing |
| ONNX export | `inference/export/export_onnx.py` | Via torch.onnx.export |
| YAML configs | `configs/` | Base + 3 experiments + 3 model sizes |
| SLURM job script | `jobs/run_test.sh` | Isambard-AI GH200 |
| Teacher verification | `jobs/check_teachers.py` | Pre-flight checkpoint + import checks |
| Unit tests | `tests/` | Forward pass, metric scale, temporal consistency, realtime engine |

### 7.2 Stubs / To Be Implemented

| Component | Location | Status |
|-----------|----------|--------|

| Metric3D V2 teacher | `models/teachers/metric3d_v2.py` | `NotImplementedError` |
| Inference demo | `inference/demo.py` | `NotImplementedError` |
| CoreML export | `inference/export/export_coreml.py` | `NotImplementedError` |
| TensorRT export | `inference/export/export_tensorrt.py` | `NotImplementedError` |
| Benchmark main() | `evaluation/benchmark.py` | Framework exists, `main()` not wired |
| Clean baseline protocol | `evaluation/protocols/clean_baseline.py` | Stub |
| Degradation sweep protocol | `evaluation/protocols/per_degradation.py` | Stub |
| iPhone LiDAR evaluation | `evaluation/protocols/iphone_lidar_eval.py` | Stub |
| Optical-flow temporal error | `evaluation/metrics/temporal_metrics.py` | Stub (flicker score implemented) |
| Benchmark callback | `training/callbacks/benchmark_callback.py` | Stub |
| Stateful Mamba hidden state passing | `inference/realtime_engine.py` | TODO (currently stateless) |

### 7.3 Legacy Code (from prior focus-map paradigm)

| Component | Location | Status |
|-----------|----------|--------|
| FocusLoss (L1+SSIM+grad+TGM for focus maps) | `utils/loss.py` | Dead code — not referenced by current training |
| Focus-map metrics (weighted F, TSS) | `utils/metrics.py` | Partially relevant (TSS is general) |
| CoC focus dataset | `dataloader/focus_dataset.py` | Functional but unused in current pipeline |
| Wild evaluation | `test_wild.py` | Interprets depth output as focus map — needs updating |

---

## 8) Model Variants

| Variant | Config | `embed_dim` | `encoder_depths` | `d_state` |
|---------|--------|-------------|-----------------|-----------|
| Small | `configs/model/mamba_depth_small.yaml` | 64 | [2, 2, 2, 2] | 16 |
| **Base** (default) | `configs/model/mamba_depth_base.yaml` | 96 | [2, 2, 4, 2] | 16 |
| Large | `configs/model/mamba_depth_large.yaml` | 128 | [2, 2, 6, 2] | 32 |

All variants share: `patch_size=4`, `t_patch=2`, `d_conv=4`, `expand=2`.

---

## 9) Suggested Paper Contributions (wording draft)

1. **Multi-teacher depth distillation** — fusing complementary image and video depth teachers (DA3, Depth Pro, VDA) into a single lightweight student via confidence-weighted ensemble supervision.
2. **Spatiotemporal SSM architecture for video depth** — alternating 4-way cross-scan spatial Mamba and causal temporal Mamba blocks in a multi-scale encoder-decoder, achieving temporal consistency without self-attention.
3. **Degradation-robust training** — a physically motivated degradation curriculum (low-light noise, motion blur) that separates student input quality from teacher supervision quality.
4. **Efficient deployment** — the student model is orders of magnitude smaller than the teacher ensemble, targeting real-time inference (30 FPS at 256×256).

---

## 10) Experimental Plan for Write-up

### 10.1 Main Comparisons
- FocusMamba (base) vs per-frame transformer baseline (FocusTransformer).
- FocusMamba vs individual teachers run at student resolution.
- FocusMamba vs other lightweight depth models (Depth Anything V2 small, MiDaS small).

### 10.2 Ablations
- Remove temporal Mamba blocks (frame-independent processing via `ablation_no_temporal.yaml`).
- Remove cross-scan (single raster direction only).
- Single teacher vs multi-teacher distillation.
- With vs without degradation curriculum.
- With vs without confidence-weighted ensembling.

### 10.3 Reporting
- **Depth metrics:** AbsRel, SqRel, RMSE, δ < 1.25.
- **Temporal metrics:** Flicker score, visual stability.
- **Efficiency:** Parameter count, FLOPs, FPS on GH200 / consumer GPU.
- **Qualitative:** Temporal sequences showing depth stability, degradation robustness comparisons.

---

## 11) Reproducibility Checklist

- Keep YAML config frozen per experiment (config snapshot saved in every checkpoint).
- Record commit hash for each run.
- Log train/val splits and random seed (seed=42 default).
- Save `best.pt` and `latest.pt` with full optimiser + scheduler state.
- Export TensorBoard scalars to CSV for plotting.
- Vendored teacher code ensures reproducibility without external git repos.

---

## 12) Quick Links to Current Implementation

**Student model:**
- `models/focus_mamba.py` — top-level model wrapper
- `models/encoder.py` — TubeletEmbedding + multi-scale encoder
- `models/decoder.py` — UNet decoder + metric depth head
- `models/mamba_block.py` — cross-scan Mamba and temporal Mamba blocks
- `models/focus_transformer.py` — transformer ablation baseline

**Teachers:**
- `models/teachers/depth_anything_v3.py` — DA3 wrapper
- `models/teachers/depth_pro.py` — Depth Pro wrapper
- `models/teachers/video_teacher.py` — VDA wrapper
- `models/teachers/vendor/` — vendored teacher architectures

**Training:**
- `train.py` — primary production training entry point
- `test_training.py` — lightweight trial script (YouTube-VOS + L1, for rapid iteration)
- `training/losses/` — SI-log, gradient, temporal, distillation, combined loss
- `training/curriculum.py` — degradation curriculum
- `training/ema.py` — EMA model

**Data:**
- `dataloader/youtube_vos.py` — YouTube-VOS clip loader
- `dataloader/degradation.py` — low-light degradation pipeline
- `dataloader/focus_dataset.py` — legacy CoC focus dataset

**Evaluation + inference:**
- `evaluation/metrics/depth_metrics.py` — depth evaluation metrics
- `inference/realtime_engine.py` — real-time inference engine
- `inference/autofocus_interface.py` — depth → focus distance

**Config:**
- `config.yaml` — legacy root config
- `configs/base.yaml` — canonical base config
- `configs/experiments/trial_youtube_vos.yaml` — trial training config

---

## 13) Notes to Self Before Final Writing

- Replace all draft claims with measured numbers from training runs.
- Run full ablation suite (temporal on/off, cross-scan on/off, single vs multi teacher).
- Wire degradation curriculum to the data pipeline for degradation-robustness experiments.
- Add failure case analysis (thin structures, sudden depth discontinuities, textureless regions).
- Profile student FPS at multiple resolutions and compare against teachers.
- Investigate whether VDA's temporal consistency transfers to the student through distillation alone, or if explicit temporal loss is required.
