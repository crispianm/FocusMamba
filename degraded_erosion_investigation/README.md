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

- **(Phase A, done)** Span-collapse + boundary-loss signature confirmed on all three
  existing runs. See `results/phaseA_signature.csv`.
- *(Phase B / C: to be filled in as jobs complete.)*
