# ACM MM Paper Outline (Recommended 2026-03-10)

## Working Title
Focused Redirection for Robust Causal Streaming Metric Depth

## One-Sentence Claim
Metric Video Depth Anything Small can be fine-tuned for true causal cache-based streaming, and degraded-streaming error is reduced most effectively by redirecting corrupted evidence both before and during temporal state updates: a cheap per-frame channel-stat alignment prefilter plus a lightweight cached-state gate.

## Recommendation
Write the paper your current runs support, not the paper you originally hoped to have.

- Student: Video Depth Anything Small only
- Target: metric depth only
- Streaming: causal cache-based VDA streaming only
- Main method: channel-stat alignment prefilter plus lightweight state gate
- Training recipe: use the best teacher/distillation setup as a fixed recipe, not as the novelty
- Teacher comparisons: appendix or late secondary section only
- Restoration comparisons: omit unless they are nearly free to run
- Mamba route: do not make it the paper unless it clearly beats the cache-based family soon

## Why This Is The Right Paper
Current runs point to a different story than the old gate-only outline.

### Strong signal you already have
- Clean streaming fine-tuning is real and stable.
- `runs/acmmm_vda_streaming_metric_20260307_half_frames/02_stream_clean` has reached `AbsRel = 0.3232` at epoch 9 and is still improving.

### Weak signal you should not center
- Teacher-only degraded sweeps are clustered tightly.
- The current best from `runs/teacher_loss_sweep_20260308/*` is about `AbsRel = 0.6426`, with most runs in the `0.64-0.68` band.
- That is useful recipe selection, but it is not a compelling main contribution by itself.

### Stronger degraded-story signal
- Adding `stats_align` to the degraded streaming setup gives a larger gain than most teacher-loss swaps.
- `runs/focused_redirection_20260310/02_anchor_vda_stats_align` is at `AbsRel = 0.6288` by epoch 4.

### Strongest current degraded result
- The best active degraded run is `runs/focused_redirection_20260310/04_hetero_gate_stats_align`.
- It has reached `AbsRel = 0.5809` at epoch 3 of 5.
- That is the clearest evidence that the winning direction is not teacher choice alone and not gate-alone framing. It is pre-alignment plus state control.

### Important caveat
- On the strongest degraded runs, `AbsRel` is improving more cleanly than some secondary metrics such as `delta1`, validation loss, and `SI-log`.
- That means the paper should make a primary claim about degraded causal-streaming error and contamination control, then verify calibration/secondary metrics before locking the final tables.

## Paper Positioning
- Venue target: ACM Multimedia
- Primary framing: robust causal streaming metric depth under realistic low-light and blur corruption
- Secondary framing: lightweight deployment-oriented robustness improvements for cache-based streaming models
- Non-goals:
  - new backbone paper
  - Mamba paper
  - restoration paper
  - teacher-comparison paper
  - generic robustness paper

## Core Story
1. The released metric VDA-S checkpoint is not enough for robust causal streaming on its own.
2. Clean streaming fine-tuning closes a meaningful portion of the offline-to-streaming gap.
3. Under degraded inputs, changing teacher supervision alone gives only modest gains.
4. A cheap per-frame input-statistics alignment step materially improves degraded streaming.
5. A lightweight cached-state gate becomes most useful after that pre-alignment is in place.
6. The combined method gives the best degraded streaming error among your current models and is the most likely path to a defensible paper.

## Main Contribution
Do not sell this as "a gate paper" anymore. Sell it as a focused redirection paper with two lightweight pieces.

### Contribution 1: Observation-space redirection
- Apply per-frame channel-stat alignment before the student sees degraded frames.
- This is not restoration and should not be framed as restoration.
- The point is to reduce destructive distribution shift before corrupted observations enter the temporal cache.

### Contribution 2: State-space redirection
- Apply a lightweight gate to attenuate cached temporal state when current features disagree with the stored state.
- The gate is a compact cache-control mechanism, not a new temporal backbone.

### What is not the contribution
- The teacher ensemble
- The exact distillation loss variant
- Any Mamba replacement
- Heavy preprocessing

## Method Scope
- Backbone: `metric_video_depth_anything_vits.pth`
- Public streaming API:
  - `forward(frames, mode="offline" | "streaming_emulated", return_state=False)`
  - `reset_stream_state(batch_size, device)`
  - `stream_step(frame_t, state, return_gate_stats=False)`
- Streaming discipline:
  - no future frames
  - no lookahead
  - no bidirectional temporal context
- Data:
  - clean frames
  - degraded frames
  - metric depth
  - valid mask
  - optional cached teacher targets aligned to clean frames

## What Goes In The Main Paper

### Main table
- offline metric VDA-S
- training-free streaming
- clean streaming fine-tuned VDA-S
- degraded streaming baseline with the chosen teacher recipe
- degraded streaming + channel-stat alignment
- degraded streaming + channel-stat alignment + state gate
- reset every 8 frames
- reset every 16 frames
- short-memory `K=8`

### Appendix or small support table
- teacher-recipe selection:
  - single teacher
  - anchor-VDA
  - heterogeneous MTKD
  - with/without gate
- optional prefilter variants:
  - `stats_align`
  - `fast_classical`
  - `learned_affine`
  - `depthwise`

### Probably not worth a main-table slot
- frozen teacher baselines
- restoration baselines
- Mamba variants

## Paper Structure Recommendation
1. Introduction
2. Related Work
3. Problem Setting: Causal Streaming Metric Depth Under Degradation
4. Baseline: Clean Causal Streaming Fine-Tuning of Metric VDA-S
5. Why Degraded Streaming Still Fails
6. Focused Redirection
7. Experimental Setup
8. Main Results
9. Diagnostics and Discussion
10. Limitations
11. Conclusion

## Section-Level Outline

### 1. Introduction
- Motivate causal streaming metric depth as a deployment setting with persistent temporal state.
- Make the failure mechanism concrete: a bad frame does not just hurt one prediction, it contaminates future predictions through the cache.
- Preview the fix as a lightweight two-part redirection mechanism.

### 2. Related Work
- metric monocular/video depth
- causal streaming and recurrent memory
- robustness under degraded video
- lightweight normalization or calibration modules
- adaptive memory control

### 3. Problem Setting
- Define offline vs training-free streaming vs streaming fine-tuned evaluation.
- Define degraded causal streaming formally.
- Define contamination persistence and the recovery-curve diagnostic.

### 4. Baseline: Clean Causal Streaming Fine-Tuning
- Show that fine-tuning VDA-S in `streaming_emulated` mode is already nontrivial but works.
- This section uses your strong clean-stream evidence as the foundation of the paper.

### 5. Why Degraded Streaming Still Fails
- Show that degraded inputs hurt more in streaming than in offline or clean-stream settings.
- Show that teacher-loss selection alone changes the result only modestly.
- Use this section to justify why a method operating on the corrupted input/cache pathway is needed.

### 6. Focused Redirection
- 6.1 Channel-stat alignment prefilter
- 6.2 Cached-state gate
- 6.3 Why the two components are complementary
- 6.4 Runtime and parameter-cost discussion

### 7. Experimental Setup
- TartanAir v2
- 392x392
- train frames 8, val frames 16 or 32 as finalized
- 4x GH200
- bf16, batch size 1, grad accumulation 4
- AbsRel as primary selection metric
- report `delta1`, RMSE, SI-log, FDV, latency, memory

### 8. Main Results
- clean streaming baseline table
- degraded robustness ablation table
- reset/short-memory comparison
- latency and parameter overhead

### 9. Diagnostics and Discussion
- contamination recovery curve
- gate activity around corruption events
- effect of channel-stat alignment on recovery behavior
- discuss why the method improves `AbsRel` first and what calibration checks remain

### 10. Limitations
- current evidence is strongest on `AbsRel`
- degraded improvements must still be verified against all secondary metrics
- results are currently centered on TartanAir
- teacher recipe still matters for training stability, even if it is not the main novelty

### 11. Conclusion
- emphasize practical causal robustness from small, targeted controls rather than a new heavy model

## Figures And Tables

### Figures
- Figure 1: problem setup and focused-redirection method
- Figure 2: contamination recovery curve
- Figure 3: qualitative degraded streaming comparisons
- Figure 4: gate activity and prefilter effect around corruption events

### Tables
- Table 1: clean streaming baseline and gap recovery
- Table 2: degraded streaming ablation table
- Table 3: latency, memory, and parameter overhead
- Appendix table: teacher-recipe sweep

## Metrics Guidance
- Primary metric: `AbsRel`
- Required secondary metrics:
  - `delta1`
  - RMSE
  - SI-log
  - FDV
  - latency in ms/frame
  - peak memory

Do not frame the paper around a broad win until the best degraded run is validated on both `AbsRel` and at least one scale-sensitive secondary metric.

## What To Emphasize
- causal cache contamination
- lightweight robustness
- simple pre-alignment is more effective than expected
- gate helps most when it controls already-aligned inputs
- practical baseline discipline

## What To Avoid
- claiming the gate alone is the main win
- claiming teacher diversity is the main novelty
- claiming restoration-quality preprocessing
- claiming universal robustness
- letting the old Mamba/FocusMamba repo identity leak into the paper story

## Acceptance Criteria For This Paper Version
- clean streaming fine-tuning remains clearly better than training-free streaming
- degraded `stats_align + gate` beats degraded baseline and simple memory heuristics on `AbsRel`
- latency overhead stays modest enough to defend the method as lightweight
- at least one secondary metric besides `AbsRel` behaves coherently on the final chosen checkpoint

## Fallback If The Next Runs Disappoint
If `stats_align + gate` finishes without a stable multi-metric win, downgrade the paper to:

`channel-stat alignment for robust causal streaming metric depth`

In that version:
- the prefilter is the method
- the gate becomes an ablation or appendix result
- teacher sweep remains a recipe-selection appendix

Do not fall back to the old gate-only paper unless later runs produce much cleaner evidence than you have now.

## Immediate Execution Checklist
- finish the `focused_redirection_20260310` runs
- compare the best degraded checkpoint against reset-8, reset-16, and `K=8`
- generate contamination-recovery plots for:
  - degraded baseline
  - stats-align only
  - stats-align + gate
- verify whether the best checkpoint also improves at least one of:
  - `delta1`
  - RMSE
  - SI-log
- keep teacher-loss sweeps as recipe-selection evidence, not the main story
