# FocusMamba: HPC Data Generation Pipeline
## Implementation Plan for GH200 Cluster Agent
================================================

## Overview

This pipeline ingests a raw video frame dataset, enforces sharpness via
GenFocus-DeblurNet, generates temporally consistent metric depth via Video
Depth Anything (VDA), and renders photorealistic defocus sequences via
GenFocus-BokehNet. The outputs are exactly the files consumed by
`dataloader/focus_dataset.py` in the FocusMamba training plan.

### Target Output Per Scene Clip

```
dataset_root/
  {scene_id}/
    frames_sharp/        ← DeblurNet output (PNG, float16)
      frame_0000.png
      frame_0001.png
      ...
    depth_maps/          ← VDA metric depth (NPY, float32, metres)
      depth_0000.npy
      depth_0001.npy
      ...
    bokeh_renders/       ← BokehNet output per CoC param set (PNG, float16)
      set_00/
        bokeh_0000.png
        ...
      set_01/
        ...
    focus_maps/          ← GT soft focus maps (NPY, float32, [0,1])
      set_00/
        focus_0000.npy
        ...
    metadata.json        ← CoC params per set: {f, N, S_focus, max_coc}
    split.txt            ← "train" | "val" | "test"
```

---

## Environment & Hardware Assumptions

- **Node**: NVIDIA GH200 (96 GB HBM3, 72-core Arm Neoverse V2)
- **OS**: Ubuntu 22.04 LTS
- **Scheduler**: SLURM
- **Python**: 3.10 via Conda
- **Storage**: NVMe scratch at `$SCRATCH`, permanent at `$STORE`

---

## PHASE 0: Environment Setup
-------------------------------

### [ ] `pipeline/00_setup_env.sh`

This is the first script the agent should run. It is idempotent.

```bash
#!/bin/bash
#SBATCH --job-name=focusmamba_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail

CONDA_ENV=focusmamba
REPO_ROOT=$SCRATCH/focusmamba_pipeline

# --- Conda environment ---
conda create -n $CONDA_ENV python=3.10 -y
conda activate $CONDA_ENV

# Core numeric + vision
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python-headless Pillow tqdm rich

# Video Depth Anything
pip install transformers==4.40.0 accelerate safetensors timm einops

# GenFocus models (adjust to actual package names / git repos)
# If GenFocus is a private/custom repo, clone and pip install -e .
# pip install genfocus   OR:
# git clone https://github.com/<org>/GenFocus $REPO_ROOT/GenFocus
# pip install -e $REPO_ROOT/GenFocus

# Metrics / inspection
pip install matplotlib seaborn scipy

# --- Model weight downloads ---
# VDA: checkpoint downloaded lazily by transformers; pre-cache here
python -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
AutoImageProcessor.from_pretrained('depth-anything/Video-Depth-Anything-Large')
AutoModelForDepthEstimation.from_pretrained('depth-anything/Video-Depth-Anything-Large')
"

# GenFocus weights: copy from $STORE or download from project share
# Expected locations (adjust paths to match your cluster layout):
#   $STORE/weights/genfocus_deblur.pth
#   $STORE/weights/genfocus_bokeh.pth

echo "Environment setup complete."
```

---

## PHASE 1: Dataset Validation & Scene Index
---------------------------------------------

### [ ] `pipeline/01_validate_dataset.py`

Scans the raw dataset root, validates frame counts and file integrity,
builds a scene index JSON, and assigns deterministic train/val/test splits.
This must be run before any subsequent stage.

**Inputs**:
- `--raw_root`: path to raw video frames directory, structured as
  `raw_root/{scene_id}/{frame_XXXX}.{jpg|png}`
- `--out_root`: path to the processed dataset root (will be created)
- `--min_frames`: minimum number of frames for a scene to be included (default: 16)

**Logic**:

```python
# Pseudocode — the agent should expand this into a full script

import os, json, hashlib, random
from pathlib import Path

VALID_EXTS = {'.jpg', '.jpeg', '.png'}

def scan_scene(scene_dir):
    frames = sorted([
        f for f in scene_dir.iterdir()
        if f.suffix.lower() in VALID_EXTS
    ])
    return frames

def assign_splits(scene_ids, seed=42):
    """80/10/10 split at scene level — deterministic."""
    rng = random.Random(seed)
    ids = sorted(scene_ids)   # sort first for determinism
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)
    splits = {}
    for i, sid in enumerate(ids):
        if i < train_end:       splits[sid] = 'train'
        elif i < val_end:       splits[sid] = 'val'
        else:                   splits[sid] = 'test'
    return splits

# Write:
#   {out_root}/scene_index.json  — {scene_id: {frames: [...], split: "train"}}
#   {out_root}/{scene_id}/split.txt  per scene
```

**Validation checks** (skip a scene and log a warning if any fail):
- Frame count >= `min_frames`
- No corrupted images (verify with PIL `Image.verify()`)
- Consistent frame dimensions within a scene (H, W must be identical across all frames)
- No duplicate scene IDs (case-insensitive)

**Output**: `{out_root}/scene_index.json`

---

## PHASE 2: Sharpening with GenFocus-DeblurNet
------------------------------------------------

### [ ] `pipeline/02_deblur.py`

Runs GenFocus-DeblurNet on all frames of each scene.
Designed to be re-entrant: skips scenes where `frames_sharp/` already exists
and contains the expected number of PNG files.

**SLURM wrapper** (`jobs/run_deblur.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=deblur_%a
#SBATCH --array=0-{N_SCENES}%32    # throttle to 32 concurrent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

python pipeline/02_deblur.py \
    --scene_index $OUT_ROOT/scene_index.json \
    --task_id     $SLURM_ARRAY_TASK_ID \
    --out_root    $OUT_ROOT \
    --batch_size  8 \
    --weights     $STORE/weights/genfocus_deblur.pth
```

**Key implementation notes**:

- Load the full scene's frames into a RAM buffer first; the GH200's 480 GB
  unified memory means this is safe for typical scenes.
- Process in overlapping temporal batches of 16 frames (stride 8) so
  DeblurNet has temporal context at boundaries. Average overlapping predictions.
- Output: write `frames_sharp/{frame_XXXX}.png` as uint8 PNG.
  If DeblurNet outputs float32, clamp to [0,1] and convert with
  `(arr * 255).round().astype(np.uint8)` before saving.
- Log per-scene PSNR (sharp vs raw) to `{out_root}/logs/deblur_psnr.csv`
  for quality monitoring.
- On CUDA OOM: catch the exception, halve `batch_size`, retry once, then
  log the scene as failed and continue.

**Progress tracking**: write a `.done` sentinel file to
`{out_root}/{scene_id}/frames_sharp/.done` when complete. The script checks
for this file at the start and skips if present.

---

## PHASE 3: Metric Depth with Video Depth Anything
----------------------------------------------------

### [ ] `pipeline/03_depth.py`

Runs Video Depth Anything (VDA) Large on the sharpened frames to produce
temporally consistent, scale-accurate metric depth maps.

**SLURM wrapper** (`jobs/run_depth.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=depth_%a
#SBATCH --array=0-{N_SCENES}%16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
```

**Key implementation notes**:

- Use the HuggingFace `transformers` pipeline with the
  `depth-anything/Video-Depth-Anything-Large` checkpoint.
- Feed the entire scene as a sequence (VDA is designed for video; feeding
  individual frames degrades temporal consistency significantly).
- If scene length > 300 frames, split into overlapping chunks of 256 frames
  with 32-frame overlap; blend the overlap region linearly.
- Outputs: save per-frame depth as `depth_maps/depth_{XXXX}.npy`
  in float32, units assumed to be metres (VDA is metric). Confirm
  the scale by checking that the median depth across indoor scenes is
  plausible (should be ~2–5 m); log this check result.
- Also save `depth_maps/stats.json` per scene:
  `{d_min, d_max, d_median, d_p5, d_p95}` computed globally over all
  frames in the scene. The dataloader uses `d_min` / `d_max` for
  Beta distribution mapping.
- Sentinel: `{out_root}/{scene_id}/depth_maps/.done`

**Temporal consistency check**: after running VDA, compute
`mean(|depth[t+1] - depth[t]|) / mean(depth[t])` across all frame pairs.
If this ratio exceeds 0.15 for any scene, flag it in
`{out_root}/logs/depth_consistency_warnings.txt` for manual inspection.

---

## PHASE 4: CoC Parameter Sampling & BokehNet Rendering
---------------------------------------------------------

### [ ] `pipeline/04_bokeh_render.py`

For each scene, samples `N_sets` independent CoC parameter sets, then runs
GenFocus-BokehNet to render a photorealistic defocus sequence for each set.

**CoC parameter sampling** (must match the FocusMamba dataloader exactly):

```python
import numpy as np
from scipy.stats import beta as beta_dist

def sample_coc_params(depth_stats, N_sets=4, seed=None):
    """
    Returns a list of N_sets dicts: {f, N, S_focus}
    Each dict defines one independent synthetic DoF configuration.
    """
    rng = np.random.default_rng(seed)
    params = []
    d_min, d_max = depth_stats['d_min'], depth_stats['d_max']

    for _ in range(N_sets):
        # Focal length: uniform in [24mm, 85mm]
        f = rng.uniform(24e-3, 85e-3)          # metres

        # F-number: log-uniform in [1.4, 8.0]
        log_N = rng.uniform(np.log(1.4), np.log(8.0))
        N = np.exp(log_N)

        # Focus distance: Beta(2,2) mapped to [d_min, d_max]
        u = beta_dist.rvs(2, 2, random_state=rng.integers(1e9))
        S_focus = d_min + u * (d_max - d_min)

        params.append({'f': f, 'N': N, 'S_focus': S_focus})
    return params
```

**CoC computation** (per pixel, per frame):

```python
def compute_coc(depth_map, f, N, S_focus, max_coc):
    """
    depth_map: (H, W) float32, metres
    Returns: focus_map (H, W) float32 in [0,1]
             coc_map   (H, W) float32 (raw, before normalisation)
    """
    # Avoid division by zero on zero-depth pixels
    depth_safe = np.maximum(depth_map, 1e-6)

    coc = (f**2 / (N * S_focus)) * np.abs(depth_safe - S_focus) / depth_safe

    coc_norm   = np.clip(coc / max_coc, 0.0, 1.0)
    focus_map  = 1.0 - coc_norm
    return focus_map, coc
```

**IMPORTANT**: `max_coc` is NOT known until Phase 5 (the CoC distribution
inspection). Insert a placeholder value of `max_coc=0.1` for Phase 4
rendering (BokehNet uses the focus map only as a guidance signal, not
the final GT). The authoritative GT focus maps are computed in Phase 5
after `max_coc` is calibrated.

**BokehNet rendering loop**:

```python
for scene_id in assigned_scenes:
    depth_stats = load_depth_stats(scene_id)
    param_sets  = sample_coc_params(depth_stats, N_sets=4, seed=hash(scene_id))

    for set_idx, params in enumerate(param_sets):
        out_dir = out_root / scene_id / 'bokeh_renders' / f'set_{set_idx:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for t, (frame, depth) in enumerate(load_scene(scene_id)):
            focus_map_guidance, _ = compute_coc(depth, **params, max_coc=0.1)

            # BokehNet expects: sharp_frame (H,W,3) uint8, focus_map (H,W) float32
            bokeh_frame = run_bokeh_net(model, frame, focus_map_guidance)

            save_png(bokeh_frame, out_dir / f'bokeh_{t:04d}.png')

    # Save params (without max_coc — that is added in Phase 5)
    save_json(
        {'sets': param_sets},
        out_root / scene_id / 'metadata_raw.json'
    )
```

**SLURM wrapper** (`jobs/run_bokeh.sh`):

```bash
#SBATCH --job-name=bokeh_%a
#SBATCH --array=0-{N_SCENES}%16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
```

- Sentinel: `{out_root}/{scene_id}/bokeh_renders/.done`

---

## PHASE 5: CoC Distribution Calibration & GT Focus Map Generation
-------------------------------------------------------------------

This phase has two sub-steps that must run **in order**.

### [ ] `pipeline/05a_inspect_coc_distribution.py`

Computes the CoC distribution over a representative sample of the dataset
to determine the correct `max_coc` normalisation value.

This is the script referenced in `config.yaml` of the FocusMamba training plan.

**Usage**:
```bash
python pipeline/05a_inspect_coc_distribution.py \
    --out_root $OUT_ROOT \
    --sample_fraction 0.2 \
    --output_plot $OUT_ROOT/logs/coc_histogram.png \
    --output_stats $OUT_ROOT/logs/coc_stats.json
```

**Logic**:

```python
# For each sampled scene, for each param set, for each frame:
#   1. Load depth map
#   2. Compute raw CoC (before normalisation) using stored params
#   3. Collect all pixel-level CoC values into a flat array

# After collection:
#   - Plot histogram
#   - Report: p50, p90, p95, p99, max
#   - Recommend max_coc = p95 value

# Output coc_stats.json:
# {
#   "p50": 0.021,
#   "p90": 0.047,
#   "p95": 0.063,   ← use this as max_coc in config.yaml
#   "p99": 0.112,
#   "max": 0.438
# }
```

**After running this script**: update `config.yaml` with the recommended
`max_coc` value. Then proceed to 05b.

### [ ] `pipeline/05b_generate_focus_maps.py`

Now that `max_coc` is known, generates the authoritative GT focus maps
and writes the final `metadata.json` for each scene.

```bash
python pipeline/05b_generate_focus_maps.py \
    --out_root $OUT_ROOT \
    --max_coc  0.063      # from 05a output
```

**Logic**:

```python
for scene_id in all_scenes:
    raw_meta = load_json(out_root / scene_id / 'metadata_raw.json')

    for set_idx, params in enumerate(raw_meta['sets']):
        out_dir = out_root / scene_id / 'focus_maps' / f'set_{set_idx:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for t, depth in enumerate(load_depth_maps(scene_id)):
            focus_map, coc_raw = compute_coc(depth, **params, max_coc=max_coc)
            # Save as float32 NPY — the dataloader reads these directly
            np.save(out_dir / f'focus_{t:04d}.npy', focus_map.astype(np.float32))

    # Write authoritative metadata
    save_json(
        {
            'sets': [
                {**p, 'max_coc': max_coc}
                for p in raw_meta['sets']
            ]
        },
        out_root / scene_id / 'metadata.json'
    )
```

**Sanity checks per scene** (log warnings, do not abort):
- `focus_map.mean()` should be in [0.2, 0.8]. If outside this range,
  the CoC params are extreme — flag the scene.
- `focus_map.std()` should be > 0.05. If lower, the scene is nearly
  uniformly sharp or blurry — flag it.

---

## PHASE 6: Output Validation
-------------------------------

### [ ] `pipeline/06_validate_outputs.py`

Checks that every expected output file exists and is loadable.
Produces a final report before handoff to the training pipeline.

```bash
python pipeline/06_validate_outputs.py \
    --out_root $OUT_ROOT \
    --scene_index $OUT_ROOT/scene_index.json \
    --report $OUT_ROOT/logs/validation_report.json
```

**Checks per scene**:

| Check | Pass criterion |
|---|---|
| `frames_sharp/` count | == raw frame count |
| `depth_maps/` count | == raw frame count |
| `bokeh_renders/set_XX/` count | == raw frame count × N_sets |
| `focus_maps/set_XX/` count | == raw frame count × N_sets |
| `metadata.json` | valid JSON, all required keys present |
| `split.txt` | one of: train, val, test |
| Focus map range | all values in [0,1] |
| Depth map range | all values > 0 |
| Frame shape consistency | all frames same (H, W) within scene |

**Output**: `validation_report.json`:
```json
{
  "total_scenes": 1200,
  "passing_scenes": 1187,
  "failed_scenes": ["scene_042", "scene_391"],
  "failure_reasons": {
    "scene_042": "missing depth_maps: 3 frames",
    "scene_391": "focus_map out of range: min=-0.003"
  },
  "split_counts": {"train": 960, "val": 120, "test": 120},
  "total_frames": 184320,
  "total_gb_on_disk": 847.3
}
```

Failed scenes should be re-queued through the relevant pipeline stage,
not silently excluded.

---

## PHASE 7: Pipeline Orchestration
-------------------------------------

### [ ] `pipeline/run_pipeline.sh`

Master script that submits all SLURM jobs in dependency order.
The agent should run this once after verifying Phase 0 is complete.

```bash
#!/bin/bash
set -euo pipefail

export OUT_ROOT=$SCRATCH/focusmamba_dataset
export STORE=$STORE   # cluster-specific permanent store

# --- Validate dataset and build scene index ---
python pipeline/01_validate_dataset.py \
    --raw_root  $RAW_DATA_ROOT \
    --out_root  $OUT_ROOT \
    --min_frames 16

N_SCENES=$(python -c "
import json
idx = json.load(open('$OUT_ROOT/scene_index.json'))
print(len(idx) - 1)   # SLURM arrays are 0-indexed
")

# --- Deblur (depends on: validate) ---
JOB_DEBLUR=$(sbatch --parsable \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --array=0-${N_SCENES}%32 \
    jobs/run_deblur.sh)

# --- Depth (depends on: deblur) ---
JOB_DEPTH=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEBLUR \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --array=0-${N_SCENES}%16 \
    jobs/run_depth.sh)

# --- Bokeh render (depends on: depth) ---
JOB_BOKEH=$(sbatch --parsable \
    --dependency=afterok:$JOB_DEPTH \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --array=0-${N_SCENES}%16 \
    jobs/run_bokeh.sh)

# --- CoC inspection (depends on: bokeh) ---
JOB_COC=$(sbatch --parsable \
    --dependency=afterok:$JOB_BOKEH \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --ntasks=1 --gres=gpu:0 --mem=32G \
    --wrap="python pipeline/05a_inspect_coc_distribution.py \
            --out_root $OUT_ROOT \
            --output_stats $OUT_ROOT/logs/coc_stats.json")

echo "================================================================"
echo " CoC inspection job submitted: $JOB_COC"
echo " MANUAL STEP REQUIRED:"
echo "   1. Wait for job $JOB_COC to complete."
echo "   2. Read $OUT_ROOT/logs/coc_stats.json"
echo "   3. Set max_coc in config.yaml to the p95 value."
echo "   4. Run: bash pipeline/run_pipeline_phase5b_onwards.sh"
echo "================================================================"
```

### [ ] `pipeline/run_pipeline_phase5b_onwards.sh`

```bash
#!/bin/bash
# Run after manually setting max_coc
MAX_COC=${1:?"Usage: $0 <max_coc value, e.g. 0.063>"}

JOB_MAPS=$(sbatch --parsable \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --wrap="python pipeline/05b_generate_focus_maps.py \
            --out_root $OUT_ROOT --max_coc $MAX_COC")

sbatch \
    --dependency=afterok:$JOB_MAPS \
    --export=ALL,OUT_ROOT=$OUT_ROOT \
    --wrap="python pipeline/06_validate_outputs.py \
            --out_root $OUT_ROOT \
            --scene_index $OUT_ROOT/scene_index.json \
            --report $OUT_ROOT/logs/validation_report.json"
```

---

## Key Interfaces Between Pipeline and Dataloader
---------------------------------------------------

The following contracts must hold between pipeline outputs and
`dataloader/focus_dataset.py`:

| Pipeline output | Dataloader expectation |
|---|---|
| `frames_sharp/{frame_XXXX}.png` uint8 | read with `cv2.imread`, normalise to [0,1] float32 |
| `depth_maps/depth_{XXXX}.npy` float32 | loaded directly with `np.load` |
| `focus_maps/set_XX/focus_{XXXX}.npy` float32 [0,1] | loaded as GT label |
| `metadata.json` `.sets[i].{f, N, S_focus, max_coc}` | available for future conditioning |
| `split.txt` | used by `ClipSampler` to filter train/val/test |
| Consistent (H, W) across all modalities in a scene | required by dataloader collation |

**ROI box**: The dataloader's `roi` field is NOT produced by this pipeline.
It should be either:
- Hardcoded to `[0,0,1,1]` (full frame) for baseline experiments, OR
- Produced by a separate object detection/saliency pipeline run offline.
  Store as `roi.json` per scene if available.

---

## Disk Space Estimates

Assumes 1200 scenes × 128 frames average × 256×256 resolution × 4 CoC sets.

| Output type | Size per frame | Total estimate |
|---|---|---|
| Sharp frames (PNG uint8) | ~100 KB | ~15 GB |
| Depth maps (NPY float32) | ~256 KB | ~39 GB |
| Bokeh renders (PNG uint8) × 4 | ~100 KB × 4 | ~62 GB |
| Focus maps (NPY float32) × 4 | ~256 KB × 4 | ~157 GB |
| **Total** | — | **~273 GB** |

Provision at least **400 GB** on `$SCRATCH` to allow for temp files.

---

## Error Recovery Notes

- All scripts write `.done` sentinel files; re-running a stage is always safe.
- SLURM array jobs that fail mid-way can be resubmitted without
  `--dependency` — the sentinel check ensures only failed scenes re-run.
- If a BokehNet render is corrupted (all-black output, wrong shape), the
  validation script in Phase 6 will flag it; delete the `.done` sentinel
  and requeue that scene's array index.
- Depth maps with NaN or Inf pixels should be filled with the scene-level
  median before saving. Log how many pixels were patched.
