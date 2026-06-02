# Degraded-erosion investigation

**Question.** In every degraded fine-tune (local `degraded_ft_518`, HPC
`degraded_ft_vkitti_hpc`, arch-ablation `B_wdda`) the degraded-input validation
metric (`aligned_abs_rel`) is **best after one epoch** then erodes monotonically,
while clean-input val keeps improving — even though training is on degraded inputs.
Why?

## Phase A — signature from existing logs (free)

`scripts/analyze_logs.py` parses any `metrics.jsonl` and plots the per-epoch
trajectory of `aligned_abs_rel`, `pred_depth_span`, `boundary_f1`, `edge_abs_rel`
for clean vs degraded val sets. Run on the three existing runs:

```
python scripts/analyze_logs.py \
    ../../runs/local/degraded_ft_vkitti_hpc ../../runs/local/degraded_ft_518 \
    ../../runs/arch_ablation/B_wdda --labels hpc local bwdda
```

Output: `results/phaseA_signature.{csv,png}`. **Finding (all three runs):** on
degraded input across epochs, `pred_depth_span` collapses (~110→14), `boundary_f1`
halves, `edge_abs_rel` rises; on clean input span stays wide and `boundary_f1`
*rises*. The model increasingly relies on sharp high-frequency cues that blur+noise
destroy, so on degraded input it collapses to a flat, edge-poor mean-depth that the
per-clip scale-shift alignment cannot recover → `aligned_abs_rel` inflates.

## Hypotheses

- **H1 (range-collapse / detail-dependence)** — real robustness loss; later epochs
  worse across *all* severities & seeds, with span collapse. *(primary)*
- **H2 (fixed-seed / severity artifact)** — worse only at the single fixed val seed
  or only at high severity (would mean the erosion is partly an eval artifact).
- **H3 (TGM)** — the temporal term drives over-smoothing / range collapse.
- **H4 (over-training / LR)** — plain overfitting; gentler optimisation should slow it.
- **H5 (EMA)** — EMA captures a transient peak. *(weak: EMA evaluates clean and
  degraded with the same weights, so it cannot explain the divergence by itself.)*

## Phase B — checkpoint ladder + severity/seed sweep

`base_small.yaml` is a faithful small-data run (2 TartanAir envs + 2 VKITTI scenes)
that reproduces the epoch-1 peak and saves a per-epoch checkpoint ladder
(`save_every_n_epochs`, added to `train.py`). `scripts/eval_severity_sweep.py` then
evaluates each `epoch_{N}.pt` across a severity grid (0/0.25/0.5/0.75/1.0 via the
degradation `severity_scale`) × seeds (different degradation realisations on the
*same* clips), reporting `aligned_abs_rel`, raw `abs_rel`, `pred_depth_span`,
`boundary_f1`. Distinguishes H1 (universal erosion + span collapse) from H2.

## Phase C — causal ablations / fixes

Each config = `base_small.yaml` + one change (compare the degraded-val trajectory
and re-run the sweep):

| config | change | tests |
|---|---|---|
| `c1_no_ema` | `ema_decay: 0` | H5 (EMA) |
| `c2_no_tgm` | `temporal_weight: 0` | H3 (TGM) |
| `c3_low_lr` | `lr 5e-5 → 1e-5` | H4 (over-training) |
| `c4_clean_mix` | `clean_probability: 0.3` | does pure-degraded training drive it? |
| `c5_severe_curriculum` | degradation narrowed to the hard end | candidate fix: force robust features |

## Running on Isambard

```
# from /projects/b5dh/FocusMamba
bash degraded_erosion_investigation/scripts/submit_all.sh
```

Each job (1 GH200, ~30–40 min train + minutes of sweep) writes
`degraded_erosion_investigation/runs/<name>/{metrics.jsonl,sweep_vkitti.csv,sweep_tartanair.csv}`;
checkpoint ladders go to `/projects/b5dh/degraded_erosion_investigation/ckpts/<name>`
(scratch — clean up after the sweeps).

## Findings log

- **(Phase A)** Span-collapse + boundary-loss signature confirmed on all three existing
  runs. See `results/phaseA_signature.{csv,png}`.

- **(Phase B — diagnosis, jobs 4973474 + ladder sweep)** `base_small` reproduced the
  epoch-1 peak (VKITTI degraded 0.313@ep1 → 0.527@ep6, span 107→7.7). The severity×seed
  sweep (`results/phaseB_severity_sweep.png`, `runs/base_small/sweep_*.csv`) is decisive:
  - **H2 (fixed-seed/severity artifact) REJECTED** — across 3 independent degradation
    seeds the metrics are nearly identical (e.g. ep6 sev1.0: 0.508/0.503/0.491). The
    erosion is not an unlucky fixed val realisation.
  - **H1 (real, severity-dependent range collapse) CONFIRMED** — error worsens after
    ep1 at *every* severity, but the magnitude scales with severity: clean (sev0)
    aligned_abs_rel +22% (0.157→0.191) with span barely moving (31→28), vs sev1.0 +65%
    (0.30→0.50) with span collapsing 21→9.6. The model overfits to sharp clean-structure
    cues; the more degradation removes those cues, the harder the prediction collapses to
    a flat, edge-poor mean-depth. boundary_f1 *rises* on clean (sharper) but falls on
    degraded — same mechanism as Phase A. Confirmed on both VKITTI and TartanAir.

- **(Phase C — ablations, jobs 4973556–4973564)** see `results/phaseC_ablations.png`,
  `results/phaseBC_summary.csv`. Erosion (peak→ep6, VKITTI sev1.0):

  | run | ep1 | ep6 | verdict |
  |---|---:|---:|---|
  | base_small | 0.30 | 0.50 | reference |
  | **c3_low_lr (5e-5→1e-5)** | **0.24** | **0.30** | **best peak + least erosion + best span retention (29→19)** |
  | c4_clean_mix (30% clean) | 0.26 | 0.36 | partial: better peak/span, still erodes |
  | c1_no_ema | 0.39 | 0.44 | EMA is *not* the cause — erosion persists without it; EMA only smooths the ep1 peak |
  | c2_no_tgm | 0.31 | 0.43 | TGM is *not* the cause — removing it collapses span *harder* (10→6.8) |
  | c5_severe_curriculum | 0.41 | 0.50 | **backfires** — forcing harder degradation collapses span catastrophically (12→2.3), positively confirming the detail-dependence mechanism |

### Diagnosis

The epoch-1 sweet-spot is an **overfitting-driven, severity-dependent loss of degradation
robustness, manifested as predicted-depth-range collapse**. As fine-tuning proceeds the
model increasingly relies on high-frequency image detail to produce confident, wide-range,
sharp depth; on degraded input (blur+noise destroy that detail) it falls back to a flat
mean-depth that the per-clip scale-shift alignment cannot recover. It is **not** an EMA
effect, **not** a TGM effect, and **not** an alignment/fixed-seed artifact.

The dominant controllable lever is **optimisation speed**: a lower LR (c3) slows the
detail-overfit, giving both a better epoch-1 model and a much flatter trajectory (on
TartanAir the c3 trajectory is essentially flat). Some clean-input mixing (c4) helps
partially. Brute-force "more degradation in training" (c5) is counterproductive. A proper
robustness fix (beyond early-stopping at ep1, which the `aligned_abs_rel` selection metric
already does) likely needs an architectural/loss change that decouples depth structure from
input sharpness (e.g. a clean↔degraded prediction/feature-consistency objective) — i.e. the
Phase-2 architecture work, now with a concrete target: prevent range collapse under degradation.
