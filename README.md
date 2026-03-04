# FocusMamba

**Degradation-Robust Metric Video Depth Estimation via Spatiotemporal State Space Modeling**

FocusMamba estimates per-pixel metric depth from video clips under challenging imaging conditions (low-light, camera shake, sensor noise). A Mamba-based encoder-decoder alternates spatial cross-scan and temporal SSM blocks across scales, distilling knowledge from an ensemble of four complementary depth teachers.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Teacher Model Setup](#teacher-model-setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Configuration](#configuration)
- [Model Variants](#model-variants)
- [Project Structure](#project-structure)

---

## Overview

FocusMamba addresses two core challenges in monocular video depth:

1. **Degradation robustness** — teachers run on clean frames, the student learns to predict depth from synthetically degraded video (low-light, motion blur, shot noise).
2. **Temporal consistency** — spatial cross-scan Mamba blocks and a dedicated temporal SSM sequence layer jointly enforce smooth depth across frames, backed by a temporal gradient matching loss.

Training supervision combines:
- **Scale-invariant log loss** against a confidence-weighted teacher ensemble.
- **Gradient smoothness loss** penalising depth discontinuities.
- **Temporal consistency loss** penalising frame-to-frame depth oscillation.
- **Circle-of-Confusion (CoC) prior** — per-pixel focus maps synthesised from depth via a lens physics model can optionally augment supervision.

---

## Architecture

```
Video clip (B, C, T, H, W)
        │
        ▼
 TubeletEmbedding           3-D Conv, patch_size × t_patch
        │
        ▼
 Sinusoidal 3-D PE          T + H + W independent encodings
        │
        ▼
 FocusMambaEncoder ─────────────────────────────────────────
  Stage 0  (res/4,  emb=96 )  SpatialMamba × 2  + TemporalMamba × 2  ┐
  Stage 1  (res/8,  emb=192)  SpatialMamba × 2  + TemporalMamba × 2  │ skip
  Stage 2  (res/16, emb=384)  SpatialMamba × 4  + TemporalMamba × 4  │ connections
  Stage 3  (res/32, emb=768)  SpatialMamba × 2  + TemporalMamba × 2  ┘
        │
        ▼
 FocusMambaDecoder           Upsampling + cross-scan Mamba + 3-D smooth conv
        │
        ▼
 Metric Depth Head           log-depth → exp → (B, 1, T, H, W) in metres
```

**SpatialMamba** sweeps tokens in four directions (left→right, right→left, top→bottom, bottom→top) within each frame.  
**TemporalMamba** scans the time axis independently per spatial location.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/crispianm/FocusMamba.git
cd FocusMamba

# 2. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Make sure CUDA toolkit tools are available (nvcc is required for mamba-ssm build)
nvcc --version
echo "$CUDA_HOME"

# 4. Create/update the environment
uv sync

# Optional helper (includes nvcc checks):
bash tools/install_mamba_deps.sh
```

> **Important:** `mamba-ssm` and `causal-conv1d` are installed as dependencies (not vendored).
> They require CUDA build tooling. If `uv sync` fails with `nvcc was not found`, load your CUDA
> module/toolkit first (for example on HPC: `module load cuda`), then re-run `uv sync`.

---

## Teacher Model Setup

Clone teacher code repositories into `./teachers`:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git    teachers/Depth-Anything-3
git clone https://github.com/DepthAnything/Video-Depth-Anything.git  teachers/Video-Depth-Anything
git clone https://github.com/apple/ml-depth-pro.git                  teachers/ml-depth-pro
```

Download teacher weights into `./checkpoints`:

| File | Teacher | Source |
|------|---------|--------|
| `depth_pro.pt` | Apple Depth Pro | https://github.com/apple/ml-depth-pro |
| `metric_video_depth_anything_vitl.pth` | Video Depth Anything | https://huggingface.co/depth-anything/Video-Depth-Anything |
| `da3_metric.safetensors` + `config.json` | Depth Anything V3 (metric-large) | https://huggingface.co/ByteDance-Seed/Depth-Anything-3 |

Verify all teachers load correctly before training:

```bash
python jobs/check_teachers.py --dry-run
```

---

## Dataset

FocusMamba trains on [YouTube-VOS](https://youtube-vos.org/). Download and place under `data/youtube-vos/` (or set `data_root` in your config):

```
data/
└── youtube-vos/
    ├── train_all_frames/
    └── test_all_frames/
```

Helper script:

```bash
bash tools/download_datasets.sh
```

Expected scene structure (for the custom depth dataset):

```
<data_root>/
└── <scene_name>/
    ├── frames_sharp/frame_XXXX.png
    ├── depth_maps/depth_XXXX.npz      # key: 'depth'
    └── focus_maps/set_XX/focus_XXXX.npz   # optional; synthesised on-the-fly otherwise
```

If precomputed focus maps are absent, they are synthesised at load time using the Circle-of-Confusion model. Calibrate `max_coc` for your dataset with:

```bash
python dataloader/inspect_coc_distribution.py --data_root <path>
```

---

## Training

### Distillation (recommended)

```bash
python train.py --config configs/experiments/distill_v1.yaml
```

### Resume from checkpoint

```bash
python train.py --config configs/base.yaml --resume checkpoints/latest.pt
```

### Key training features

| Feature | Details |
|---------|---------|
| Mixed precision | `bf16` autocast + GradScaler |
| Gradient clipping | Configurable; default `0.5` |
| LR schedule | Linear warmup → cosine decay |
| EMA | Exponential moving average of student weights |
| Degradation curriculum | Progressively harder degradation over training |
| TensorBoard | Losses, metrics, depth visualisations |
| Checkpointing | Best model saved by validation AbsRel |

Monitor training:

```bash
tensorboard --logdir runs/
```

---

## Evaluation

```bash
python -m evaluation.benchmark \
    --config configs/base.yaml \
    --checkpoint checkpoints/best.pt
```

Metrics reported: **AbsRel**, **SqRel**, **RMSE**, **δ < 1.25**, **SSIM**, **PSNR**, **Temporal Stability Score (TSS)**.

Baseline comparisons are in `evaluation/baselines/`:

```bash
python evaluation/baselines/run_depth_anything.py --config configs/base.yaml
python evaluation/baselines/run_metric3d.py        --config configs/base.yaml
```

---

## Inference

### Real-time demo (webcam or video file)

```bash
python -m inference.demo --checkpoint checkpoints/best.pt --source webcam
python -m inference.demo --checkpoint checkpoints/best.pt --source path/to/video.mp4
```

### Autofocus interface

`inference/autofocus_interface.py` wraps the model as an autofocus signal source, outputting the estimated focal distance for camera control pipelines.

### Export

ONNX / TorchScript export utilities are in `inference/export/`.

---

## Configuration

All hyperparameters are controlled via YAML. The base config is [`config.yaml`](config.yaml); experiment-specific overrides live in [`configs/experiments/`](configs/experiments/).

Key fields:

```yaml
# Model
model_type: mamba          # 'mamba' or 'transformer'
embed_dim: 96
encoder_depths: [2, 2, 4, 2]

# Data
window_length: 8           # frames per clip
resolution: [256, 256]

# Distillation
distillation:
  enabled: true
  teachers: [depth_anything_v3, depth_pro, metric3d_v2, video_depth_anything]

# Degradation simulation
degradation:
  enabled: false
  lux_level: 10.0
  curriculum: true          # enable in experiments/distill_v1.yaml
```

---

## Model Variants

| Variant | Config | `embed_dim` | `encoder_depths` | `d_state` |
|---------|--------|-------------|-----------------|-----------|
| Small | `configs/model/mamba_depth_small.yaml` | 64 | [2, 2, 2, 2] | 16 |
| **Base** (default) | `configs/model/mamba_depth_base.yaml` | 96 | [2, 2, 4, 2] | 16 |
| Large | `configs/model/mamba_depth_large.yaml` | 128 | [2, 2, 6, 2] | 32 |

---

## Project Structure

```
FocusMamba/
├── models/                  # Model definitions (encoder, decoder, mamba/transformer blocks)
├── training/                # Trainer, losses, curriculum, EMA, callbacks
├── dataloader/              # Dataset, CoC synthesis, degradation pipeline, YouTube-VOS loader
├── evaluation/              # Benchmark runner, metrics, baselines, visualisation
├── inference/               # Demo, autofocus interface, realtime engine, export
├── configs/                 # YAML configs (base + model variants + experiments)
├── checkpoints/             # Teacher weights (download separately)
├── teachers/                # Teacher model code (clone separately)
├── tools/                   # Profiling, visualisation, dataset utilities
├── tests/                   # Unit tests (pytest)
├── train.py                 # Main training entry point
├── config.yaml              # Default configuration
└── requirements.txt
```

---

## Citation

> Working title: *FocusMamba: ROI-Conditioned Spatiotemporal State Space Modeling for Video Focus Map Prediction*  
> (manuscript in preparation)
