# ACM MM 2026 Live Paper Record

Last updated: 2026-03-12

## Working Title
Lightweight Degradation-Adaptive Streaming Video Depth Anything for Metric Depth

## Problem Statement
Pretrained Video Depth Anything (VDA) fails badly on degraded video. In the target setting we care about, degraded frames enter a causal streaming pipeline directly, so we cannot assume a separate denoiser or restoration model runs first. The paper goal is therefore a lightweight streaming metric-depth model that improves degraded-data performance first, while preserving practical speed.

## Locked Evaluation Priorities
- Primary metrics: `AbsRel`, `delta1`
- Model-selection rule: prefer degraded-data performance only
- Scientific rule: do not compare capped subset screens against full-data validation as final evidence

## Current Evidence Snapshot (2026-03-12)

### Verified repo state
- Degradation pipeline lives in `dataloader/degradation.py`
- Historical archive lives in `oldruns/`
- Existing lightweight robustness hooks already in code:
  - input prefilters in `models/video_depth_anything_model.py`
  - cache gating in `models/teachers/vendor/video_depth_anything/dpt_temporal.py`
- Validation baseline directory `runs/vda_validation_baselines/` exists but still needs matched official-VDA evaluation outputs

### Degradations currently implemented
- exposure drop
- anisotropic blur
- shot noise
- read noise
- banding / fixed-pattern noise
- quantization
- linear-RGB processing

### Degradations currently active in legacy training configs
- exposure drop
- anisotropic blur
- shot noise
- read noise
- temporal jitter

### Best verified runs so far
- Best clean causal stream:
  - `runs/acmmm_vda_streaming_metric_20260307_half_frames/02_stream_clean`
  - `AbsRel 0.31894`, `delta1 0.8304`
  - data budget: full train / full val
- Best plain degraded streaming VDA-S baseline:
  - `runs/teacher_loss_sweep_20260308/04_teacher_anchor_vda`
  - `AbsRel 0.61939`, `delta1 0.6635`
  - data budget: full train / full val
- Best degraded modified small-model result by `AbsRel`:
  - `runs/focused_redirection_paperlock_20260310/20_hetero_stats_align_neutral_stats`
  - `AbsRel 0.56787`, `delta1 0.4011`
  - data budget: `max_train_trajectories=128`, `max_val_trajectories=16`
- Best degraded balanced gate-inclusive candidate:
  - `runs/focused_redirection_paperlock_20260310/17_hetero_stats_align_gate_curriculum_mixed`
  - `AbsRel 0.61643`, `delta1 0.6216`
  - data budget: `128 / 16`

### Current interpretation
- `stats_align` is real and consistently useful
- gate-only is not enough
- the current best `AbsRel` run is a stats-alignment variant without gate
- gate remains promising only when paired with a stronger front-end correction strategy

### Comparability caveat
Focused-redirection paper-lock runs used capped subset budgets (`128 / 16`), while the strongest teacher-loss baselines used broader/full data. These are useful for screening, but not scientifically valid final rankings. Final claims must come from matched full-validation reruns.

## Scientific Validity Rules
- Use the current validation split, but evaluate finalists on full validation unless a screen is explicitly marked as subset-only.
- Report both `AbsRel` and `delta1` for every finalist.
- Use per-clip CSV outputs for all finalist checkpoints.
- Use seed averaging for final paper models.
- Use bootstrap confidence intervals on per-clip metrics for final comparisons.
- Treat teacher recipe as a controlled training choice, not the main paper novelty.

## Active Hypothesis
Plain degraded-data fine-tuning is probably not enough for ACM MM 2026. The strongest paper direction is a lightweight architectural extension of streaming VDA-S that adapts corrupted low-level input or pre-temporal features before degradation contaminates temporal state.

## Main Technical Direction
Keep the VDA-S backbone and streaming pipeline mostly unchanged. Add a very small degradation-adaptation module at the front of the model, and test a second lightweight adapter at the first two temporal-entry stages.

## Active Architecture Queue

### Rank 1: `stats_guided_front_adapter` (main candidate)
- Location: before ImageNet normalization in `models/video_depth_anything_model.py`
- Structure:
  - optional `stats_align`
  - two identity-initialized residual blocks
  - `1x1 3->16`, `DWConv 3x3`, `GELU`, `1x1 16->3`
  - small stats-conditioned channel modulation from per-frame RGB mean/std
- Why it is first:
  - strongest prior evidence already favors observation-space correction
  - lowest-risk learned extension of `stats_align`
  - minimal compute overhead

### Rank 2: `pre_temporal_stage_adapter` (main candidate)
- Location: immediately before temporal mixing at `layer3` and `layer4` in `models/teachers/vendor/video_depth_anything/dpt_temporal.py`
- Structure:
  - residual bottleneck adapter
  - `1x1 C->C/4`, `DWConv 3x3`, `GELU`, `1x1 C/4->C`
  - zero-initialized residual path
- Why it is second:
  - directly targets corrupted features before they enter the temporal cache
  - still lightweight
  - more invasive than the front adapter, so it should be tested after the front-end path

### Rank 3: `front_adapter + existing_cache_gate` (main combo candidate)
- Keep only if it beats the front adapter alone under matched full validation.

### Rank 4: tiny local-attention front adapter (ablation only)
- Only test if the conv-first front adapter plateaus and the runtime cost stays small.

## Implementation Status
- Added `stats_guided_front_adapter` plumbing to `models/video_depth_anything_model.py`
- Added `pre_temporal_stage_adapter` plumbing to `models/teachers/vendor/video_depth_anything/dpt_temporal.py`
- Added full-validation checkpoint evaluator: `tools/evaluate_checkpoint.py`
- Extended run aggregation so it now records:
  - prefilter type
  - temporal-adapter type
  - train/val budgets
  - subset vs full budget scope
- Extended `tools/generate_focused_redirection_suite.py` with a dedicated `front_adapter_stageb` screen suite

## Experimental Plan

### Stage A: Scientific validity lock
- Run official pretrained VDA validation sweeps into `runs/vda_validation_baselines/`
- Re-evaluate these checkpoints on identical full validation using `tools/evaluate_checkpoint.py`:
  - `teacher_anchor_vda`
  - `teacher_hetero_mtkd_plus_gate`
  - `anchor_stats_align_gate`
  - `hetero_gate_stats_align`
  - `hetero_stats_align_neutral_stats`
  - `hetero_stats_align_clean_consistency`
  - `hetero_stats_align_gate_curriculum_mixed`
  - `vdal_degraded_stream`

### Stage B: First architecture screen
- Fixed screen budget:
  - `max_train_trajectories=128`
  - `max_val_trajectories=16`
  - `max_epochs=6`
- Fixed recipe:
  - `anchor_vda`
  - current CombinedLoss weights unchanged
- First 8 runs:
  - baseline
  - `stats_align`
  - `stats_guided_front_adapter`
  - `stats_guided_front_adapter + gate`
  - `pre_temporal_stage_adapter`
  - `pre_temporal_stage_adapter + stats_align`
  - `front_adapter + clean_consistency`
  - `front_adapter + mixed curriculum`

### Stage C: Follow-up architecture sweep
- Promote only the top two Stage-B adapter families
- Confirm under `heterogeneous_mtkd`
- Add neutral-stats, bright-stats, and gate variants only for promoted families

### Stage D: Full-data promotion
- Promote top 4 small-model candidates plus:
  - `teacher_anchor_vda`
  - `stats_align`
- Run full validation on the current split
- Run 3 seeds for the final top 2 learned candidates plus the strongest prior baseline

### Stage E: Diagnostics and deployment evidence
- Use `evaluation/contamination_recovery.py`
- Measure params and FPS for every finalist
- Reject finalists that break the lightweight deployment constraint

## Decision Logic
- Plain fine-tuning is insufficient if the best no-architecture baseline loses by at least `0.015 AbsRel` after matched full validation and does not recover the gap on `delta1`.
- Keep a new module only if it beats both:
  - matched plain degraded baseline
  - matched `stats_align` baseline
  on at least one primary metric, while not losing the other primary metric by more than `0.01`.
- Stop investing in an architecture after subset screening if it trails matched `anchor + stats_align` by more than:
  - `0.03 AbsRel`
  - `0.02 delta1`
  or costs more than `15%` FPS for no real gain.
- Gate stays in the main method only if `front_adapter + gate` beats `front_adapter` alone under the same matched evaluation.

## Open Questions
- Does the `0.56787` neutral-stats win survive matched full validation?
- Does gate help after matched reruns and seed averaging?
- Is a tiny learned front adapter enough to beat plain `stats_align` with a publishable novelty margin?
- How much of the improvement is truly architectural rather than validation-budget artefact?

## Running Evidence Table

| run | train budget | val budget | best epoch | AbsRel | delta1 | FPS | params | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `acmmm_vda_streaming_metric_20260307_half_frames/02_stream_clean` | full | full | - | 0.31894 | 0.8304 | - | 29.0M | strongest clean causal stream |
| `teacher_loss_sweep_20260308/04_teacher_anchor_vda` | full | full | - | 0.61939 | 0.6635 | - | 29.0M | strongest plain degraded baseline |
| `focused_redirection_paperlock_20260310/20_hetero_stats_align_neutral_stats` | 128 | 16 | - | 0.56787 | 0.4011 | 51.9 | 29.0M | best degraded `AbsRel`, subset-only evidence |
| `focused_redirection_paperlock_20260310/17_hetero_stats_align_gate_curriculum_mixed` | 128 | 16 | - | 0.61643 | 0.6216 | 48.8 | 29.1M | best balanced gate-inclusive candidate so far |
| `focused_redirection_paperlock_20260310/03_anchor_stats_align_gate` | 128 | 16 | - | 0.58085 | 0.4457 | 49.1 | 29.1M | strongest anchor recipe with gate |
| `focused_redirection_20260310/05_vdal_degraded_stream` | 128 | 16 | - | 0.58730 | 0.6048 | 28.8 | 384.4M | reference only, not lightweight |

## Codex-Ready Next Steps
1. Run matched full-validation reruns with `tools/evaluate_checkpoint.py`.
2. Generate the Stage-B first-wave screen with:
   - `python tools/generate_focused_redirection_suite.py --suite-kind front_adapter_stageb`
3. Launch:
   - baseline
   - `stats_align`
   - `front_adapter`
   - `front_adapter + gate`
4. Promote only if the front adapter produces a real win under the locked primary metrics.
