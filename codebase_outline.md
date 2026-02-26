# Codebase Outline: Degradation-Robust Video Depth Estimation for Cinematic Autofocus

## Purpose of This Document
This document describes the complete codebase required to train, evaluate, and benchmark a
degradation-robust metric video depth estimation model targeting real-time cinematic autofocus.
Use it to: (1) audit existing code and archive anything that doesn't map to a module below,
(2) identify gaps where new files must be created, and (3) understand how all components connect.

The target venues are **ACM Multimedia 2026** or **SIGGRAPH 2026**.

---

## Project Root Structure

```
project/
├── configs/                  # All YAML experiment configs
├── data/                     # Dataset classes, degradation pipeline, preprocessing
├── models/                   # Architecture definitions
├── training/                 # Training loop, distillation losses, schedulers
├── evaluation/               # Benchmark suite, metrics, visualisation
├── inference/                # Real-time inference, export, camera integration
├── tools/                    # One-off scripts: data prep, teacher extraction, profiling
├── tests/                    # Unit and integration tests
├── docs/                     # Paper figures, architecture diagrams
├── requirements.txt
├── setup.py
└── README.md
```

---

## 1. `configs/`

All training runs are fully reproducible from a single YAML. No hardcoded hyperparameters anywhere else.

```
configs/
├── base.yaml                 # Shared defaults (optimizer, scheduler, logging)
├── model/
│   ├── mamba_depth_small.yaml
│   ├── mamba_depth_base.yaml
│   └── mamba_depth_large.yaml
├── data/
│   ├── synthetic_degradation.yaml   # Degradation types, severity ranges, probabilities
│   ├── iphone_lidar.yaml            # iPhone 17 Pro LiDAR eval set config
│   └── teachers.yaml                # Paths and settings for each teacher model
└── experiments/
    ├── distill_v1.yaml
    └── ablation_no_temporal.yaml    # Frame-by-frame baseline ablation
```

**Key config fields to define:**
- `degradation.types`: list of active degradation types (noise, blur, rain, haze, fog, compression)
- `degradation.severity_range`: per-type min/max severity for curriculum
- `model.temporal_window`: number of frames the model sees at once
- `distillation.teachers`: list of teacher names, weights, and which loss each contributes to
- `inference.target_fps`: used to gate architecture search and profiling

---

## 2. `data/`

### 2.1 Source Datasets

The model trains on existing video depth datasets with synthetic degradation applied on-the-fly.
No new labeled training data is required — degradation is synthesised during loading.

**Recommended source datasets (to download separately):**
- **DDAD** (Dense Depth for Autonomous Driving) — metric depth, outdoor video
- **NYUv2** — indoor metric depth, image-level (used for image teachers)
- **ScanNet** — indoor video with depth
- **DIML Outdoor** — natural outdoor scenes
- **DAVIS-Blur** — clean natural video (depth from teacher pseudo-labels)

```
data/
├── datasets/
│   ├── base_dataset.py           # Abstract base: returns (video_clip, depth_map, metadata)
│   ├── ddad_dataset.py
│   ├── nyu_dataset.py
│   ├── scannet_dataset.py
│   ├── davis_dataset.py          # Pseudo-labeled by teacher at preprocessing time
│   └── iphone_lidar_dataset.py   # Eval only. Handles .depth/.rgb paired Apple format
├── degradation/
│   ├── pipeline.py               # Composable degradation pipeline, applied per-clip
│   ├── noise.py                  # Gaussian, Poisson (shot noise), ISO-correlated noise model
│   ├── blur.py                   # Gaussian blur, defocus PSF, motion blur
│   ├── atmospheric.py            # Rain streaks, haze (Koschmieder model), fog, smoke
│   ├── compression.py            # JPEG, H.264/H.265 codec artifacts (via ffmpeg subprocess)
│   └── lens.py                   # Vignetting, chromatic aberration, lens flare
├── preprocessing/
│   ├── extract_pseudo_labels.py  # Run teacher models over DAVIS; save .npy depth maps
│   ├── align_iphone_depth.py     # Spatiotemporal alignment of LiDAR + RGB from iPhone
│   ├── compute_dataset_stats.py  # Mean/std per dataset for normalization
│   └── verify_dataset.py        # Sanity-check all splits before training
├── transforms.py                 # Shared geometric augmentations (crop, flip, resize)
├── collate.py                    # Batch collation for variable-length clips
└── datamodule.py                 # PyTorch Lightning DataModule wrapping all datasets
```

### 2.2 Degradation Pipeline Design

The pipeline is stochastic and applied **per training clip at load time** (not pre-computed),
ensuring the model never sees the same degradation twice. Design principles:

- Each degradation is a stateless callable `(frames: Tensor, rng: Generator) -> Tensor`
- Severity is sampled from a per-type range defined in config
- A **curriculum scheduler** (in `training/`) increases severity over training
- Degradations are applied in a randomised order to avoid order-bias
- A **clean branch** is preserved at all times for computing distillation targets

IMPORTANT: Leave this pipeline to be implemented by the user, add files where they should be but do not fill them in with any degradations, just list the classes/functions and what their expected inputs and outputs should be (with shapes and scales i.e. [B,C,H,W] and [0,1] or [-1,1]).

---

## 3. `models/`

```
models/
├── depth_model.py            # Top-level nn.Module: assembles encoder + temporal + decoder
├── encoders/
│   ├── mamba_encoder.py      # Mamba2 (SSM) based visual encoder, 2D spatial + 1D temporal
│   ├── conv_encoder.py       # Fallback CNN encoder for ablation
│   └── pretrained_init.py    # Load ImageNet or VideoMAE pretrained weights into encoder
├── temporal/
│   ├── temporal_fusion.py    # Causal temporal state passing between frames
│   └── positional.py        # Temporal positional encoding
├── decoder/
│   ├── metric_head.py        # Outputs metric depth (not relative); scale via scene prior
│   ├── uncertainty_head.py   # Optional: per-pixel confidence, useful for AF pipeline
│   └── upsample.py          # Pixel shuffle + bilinear hybrid upsampling
└── teachers/
    ├── teacher_base.py       # Common interface: takes clean frame(s), returns depth
    ├── depth_anything_v3.py  # Image teacher wrapper
    ├── depth-pro.py          # Image/video teacher wrapper
    ├── metric3d_v2.py        # Metric image teacher wrapper
    └── video_teacher.py      # Placeholder for video-specialised teacher (e.g. VideoDepthAnything)
```

### 3.1 Architecture Notes

**Why Mamba for this task:**
- Processes video as a sequence; temporal state is propagated causally (frame N sees frames 0..N-1)
- Linear time complexity in sequence length → achievable real-time at 1080p
- Unlike transformers, does not require quadratic attention over all frame pairs

**Metric depth output:**
- The decoder predicts log-depth, exponentiated to enforce positivity
- Scale anchoring: a learnable scene-type prior (indoor/outdoor/cinematic) shifts the absolute scale
- Do NOT use affine-invariant depth for this task; autofocus requires metric accuracy

**Uncertainty head** (recommended for paper, not optional for AF use):
- Predicts per-pixel aleatoric uncertainty alongside depth
- Allows the AF system to fall back gracefully when depth is unreliable
- Adds a small NLL loss term during training

---

## 4. `training/`

```
training/
├── trainer.py                # PyTorch Lightning LightningModule: full train/val step
├── losses/
│   ├── distillation.py       # Core multi-teacher distillation loss
│   ├── scale_invariant.py    # SI-log depth loss (metric)
│   ├── gradient.py           # Edge-aware depth smoothness loss (leave to the user to implement)
│   ├── temporal.py           # Temporal consistency loss (leave to the user to implement)
│   └── combined.py           # Weighted sum of all losses; weights from config
├── curriculum.py             # Degrades severity schedule; interfaces with data pipeline
├── ema.py                    # Exponential moving average of student weights
└── callbacks/
    ├── visualise_depth.py    # Logs colourised depth predictions to W&B / TensorBoard
    ├── benchmark_callback.py # Runs eval benchmark every N epochs, logs AbsRel/δ1
    └── latency_profiler.py   # Measures fps at each checkpoint; flags regressions
```

### 4.1 Distillation Loss Design

This is the most novel training component and should be described carefully in the paper.

```python
# Conceptual structure — implement in losses/distillation.py
#
# For each training sample:
#   1. Run student on DEGRADED clip   -> student_depth
#   2. Run each teacher on CLEAN clip -> teacher_depths (no_grad)
#   3. Compute per-teacher loss:
#       image_teacher_loss  = SI_log(student_depth, image_teacher_depth)
#       video_teacher_loss  = SI_log(student_depth, video_teacher_depth)
#                           + temporal_consistency(student_depth, video_teacher_depth)
#   4. Ensemble teacher targets via learned or fixed weighting
#   5. Optional: confidence-weighted loss (down-weight teacher outputs where
#      teachers themselves are uncertain — use teacher ensemble disagreement as proxy)
```

**Key design decision to justify in paper:** teachers are run on the *clean* version of the clip.
The student must learn to produce clean-equivalent depth from degraded input. This is the core
supervised signal. If teachers were run on degraded input, they would propagate degradation errors.

---

## 5. `evaluation/`

This section defines the **benchmark**, which is itself a paper contribution.

```
evaluation/
├── benchmark.py              # Main entry point: runs all eval protocols, outputs JSON report
├── metrics/
│   ├── depth_metrics.py      # AbsRel, SqRel, RMSE, δ1/δ2/δ3, SI-log
│   ├── temporal_metrics.py   # Temporal consistency error (OPW), flicker score
│   └── autofocus_metrics.py  # Focus accuracy (distance error), focus latency, pull success rate
├── protocols/
│   ├── per_degradation.py    # Sweep over all degradation types at 3 severity levels
│   ├── clean_baseline.py     # Performance on clean input (regression check)
│   ├── iphone_lidar_eval.py  # Ground-truth eval on your captured LiDAR dataset
│   └── latency_eval.py       # FPS measurement on target hardware configs
├── baselines/
│   ├── run_depth_anything.py
│   ├── run_unidepth.py
│   ├── run_metric3d.py
│   └── run_framewise_robust.py  # Image model applied frame-by-frame (the key ablation)
└── visualise/
    ├── error_maps.py
    ├── degradation_sweep_plot.py  # Performance vs. severity curves (paper Figure 3 or 4)
    └── temporal_comparison.mp4_generator.py  # Side-by-side video for supplementary material
```

---

## 6. `inference/`

```
inference/
├── realtime_engine.py        # Stateful inference class; maintains Mamba hidden state across frames
├── export/
│   ├── export_onnx.py
│   ├── export_tensorrt.py    # TRT optimisation for NVIDIA Jetson / camera workstations
│   └── export_coreml.py      # For potential on-device iPhone deployment / demo
├── autofocus_interface.py    # Takes depth map + subject bounding box → focus distance (mm)
└── demo.py                   # Webcam / video file demo with live depth overlay
```

### 6.1 Stateful Inference

Unlike training (which processes fixed clips), real-time inference must be **causal and online**:
- The Mamba state is maintained across frames; never reset mid-shot
- A shot-cut detector triggers a state reset to prevent cross-scene contamination
- Target: ≤ 33ms per frame (30fps) on an RTX 4080 or equivalent; ≤ 16ms for 60fps

---

## 7. `tools/`

One-off utility scripts. Not part of the training loop.

```
tools/
├── download_datasets.sh          # Download scripts for DDAD, NYUv2, ScanNet, DAVIS
├── profile_model.py              # FLOPs, parameter count, memory, latency table
├── visualise_degradation.py      # Generate paper figure showing degradation types + severities
├── compare_teacher_outputs.py    # Side-by-side teacher disagreement analysis
├── capture_iphone_dataset.py     # iOS shortcut / instructions for paired RGB+LiDAR capture
└── make_benchmark_report.py      # Compile eval JSONs into LaTeX table for paper
```

---

## 8. `tests/`

```
tests/
├── test_degradation_pipeline.py  # Each degradation type: output shape, value range, determinism
├── test_model_forward.py         # Forward pass on dummy input; check output shape + no NaN
├── test_temporal_consistency.py  # Feed identical frames; verify depth doesn't drift
├── test_distillation_loss.py     # Loss goes to 0 when student == teacher
├── test_metric_scale.py          # Verify output is in metric units (not relative)
└── test_realtime_engine.py       # Stateful inference: state persists, reset works correctly
```

---

## 9. Key External Dependencies

```
# Deep learning
torch >= 2.2
torchvision
pytorch-lightning >= 2.2
einops

# Mamba / SSM
mamba-ssm          # pip install mamba-ssm (requires CUDA)
causal-conv1d

# Teacher models (install metric models from their respective repos)
# - Depth Anything V3:      github.com/ByteDance-Seed/Depth-Anything-3
# - Metric3D v2:            github.com/YvanYin/Metric3D
# - Video Depth Anything:   github.com/DepthAnything/Video-Depth-Anything
# - Apple Depth Pro:        github.com/apple/ml-depth-pro

# Data / vision
opencv-python
imageio[ffmpeg]
albumentations       # Some standard augmentations
kornia               # Geometric transforms, colour space ops

# Logging & experiment tracking
tensorboard

# Export
onnx
onnxruntime

# Evaluation
pandas
matplotlib
seaborn
```

---

## 10. Data Flow Summary

```
Training:
  Source Video Clip (clean)
      │
      ├──► Degradation Pipeline ──► Degraded Clip ──► Student Model ──► Student Depth
      │                                                                        │
      └──► Teacher Models (frozen) ──► Teacher Depths ─────────────────► Distillation Loss
                                                                               │
                                                  + Temporal Consistency Loss ─┤
                                                  + SI-Log Depth Loss ─────────┘
                                                                               │
                                                                     Backprop to Student

Evaluation:
  iPhone LiDAR Clip (degraded, real)
      │
      ├──► Student Model ──► Predicted Depth ──► Depth Metrics (AbsRel, δ1, ...)
      │                                      └──► AF Metrics (FDE, PSR, FAL)
      └──► Baseline Models ──► same metrics for comparison table
```

---

## 11. What to Archive from Existing Code

When auditing existing code, **archive** (move to `archive/`) anything that:
- Implements a depth model without temporal/video awareness
- Implements depth estimation without degradation handling
- Is a training loop that doesn't support multi-teacher distillation
- Is an evaluation script that only reports AbsRel/δ1 without degradation stratification
- Is a data loader that doesn't return (clean_clip, degraded_clip, depth) triplets

**Keep and adapt** anything that:
- Loads or preprocesses any of the source datasets listed in §2.1
- Wraps any of the teacher models listed in §4.1
- Implements SI-log loss, scale-invariant loss, or gradient smoothness loss
- Implements RAFT or other optical flow estimation
- Handles video I/O (clip extraction, fps handling, frame buffering)

---

## 12. Suggested Build Order

1. `data/degradation/pipeline.py` + unit tests — validate degradation works before anything else
2. `data/datasets/` for one dataset (DDAD) end-to-end
3. `models/depth_model.py` — minimal forward pass, dummy encoder
4. `training/losses/distillation.py` — with two teachers
5. `training/trainer.py` — full train step, log losses
6. `evaluation/metrics/` + `evaluation/protocols/clean_baseline.py`
7. Swap dummy encoder for `models/encoders/mamba_encoder.py`
8. Add remaining datasets
9. `data/datasets/iphone_lidar_dataset.py` + `evaluation/protocols/iphone_lidar_eval.py`
10. `inference/realtime_engine.py` + latency profiling
11. Full benchmark sweep → paper tables and figures
