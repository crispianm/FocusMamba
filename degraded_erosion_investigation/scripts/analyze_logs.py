#!/usr/bin/env python3
"""Phase A: characterise the degraded-val epoch-1-peak / erosion from any metrics.jsonl.

Parses validation events, extracts the diagnostic trajectory for every validation
set (clean & degraded), writes a tidy CSV and a multi-panel plot. The point is to
show the *physical signature* of the erosion: predicted depth-range collapse and
boundary loss on degraded input while clean input stays sharp.

Usage:
    python analyze_logs.py RUN_DIR_OR_METRICS [RUN2 ...] \
        --labels hpc local bwdda \
        --out-dir ../results --tag phaseA_signature
"""
import argparse
import json
import os
from collections import defaultdict

# Diagnostics we track per validation set. The first is the selection metric.
DIAG = [
    "aligned_abs_rel",
    "abs_rel",
    "pred_depth_span",
    "pred_depth_p50",
    "pred_depth_p99",
    "boundary_f1",
    "edge_abs_rel",
    "fdv",
    "tgm_metric",
]


def _metrics_path(p):
    if os.path.isdir(p):
        return os.path.join(p, "metrics.jsonl")
    return p


def parse_run(path):
    """Return {val_set_name: [ {epoch, <diag>...}, ... ]} sorted by epoch.

    Pretrain validation is recorded as epoch 0.
    """
    sets = defaultdict(dict)  # name -> {epoch: {diag: val}}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ev = d.get("event")
        if ev not in ("validation", "pretrain_validation"):
            continue
        epoch = 0 if ev == "pretrain_validation" else d.get("epoch", 0)
        m = d.get("metrics", {})
        # Prefer the structured block when present.
        vsets = d.get("validation_sets")
        if isinstance(vsets, dict) and vsets:
            for name, sm in vsets.items():
                sets[name][epoch] = {k: sm.get(k) for k in DIAG}
            continue
        # Flat keys: val_<name>_<metric>. Assign each key to its LONGEST matching
        # diag suffix so e.g. val_x_aligned_abs_rel -> diag aligned_abs_rel (not abs_rel).
        diags_by_len = sorted(DIAG, key=len, reverse=True)
        for k, v in m.items():
            if not k.startswith("val_"):
                continue
            for diag in diags_by_len:
                suf = "_" + diag
                if k.endswith(suf):
                    name = k[: -len(suf)]  # keeps the leading "val_"
                    sets[name].setdefault(epoch, {})[diag] = v
                    break
    # sort by epoch
    out = {}
    for name, by_ep in sets.items():
        rows = [dict(epoch=ep, **by_ep[ep]) for ep in sorted(by_ep)]
        out[name] = rows
    return out


def write_csv(all_runs, out_csv):
    cols = ["run", "val_set", "epoch"] + DIAG
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for run_label, sets in all_runs.items():
            for name, rows in sorted(sets.items()):
                for r in rows:
                    vals = [run_label, name, str(r["epoch"])]
                    for d in DIAG:
                        v = r.get(d)
                        vals.append("" if v is None else "%.6g" % v)
                    f.write(",".join(vals) + "\n")
    print("wrote", out_csv)


def make_plot(all_runs, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable (%s); skipping plot" % e)
        return
    panels = ["aligned_abs_rel", "pred_depth_span", "boundary_f1", "edge_abs_rel"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.0 * len(panels)), sharex=True)
    for ax, metric in zip(axes, panels):
        for run_label, sets in all_runs.items():
            for name, rows in sorted(sets.items()):
                xs = [r["epoch"] for r in rows if r.get(metric) is not None]
                ys = [r[metric] for r in rows if r.get(metric) is not None]
                if not xs:
                    continue
                deg = "degraded" in name
                style = "-" if deg else "--"
                ax.plot(xs, ys, style, marker="o", ms=3,
                        label="%s/%s" % (run_label, name.replace("val_", "")))
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("epoch (0 = pretrain)")
    axes[0].set_title("Degraded (solid) vs clean (dashed): erosion signature")
    axes[0].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs or metrics.jsonl paths")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--tag", default="phaseA_signature")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(p)) for p in args.runs]
    if len(labels) != len(args.runs):
        ap.error("--labels count must match number of runs")

    all_runs = {}
    for label, p in zip(labels, args.runs):
        mp = _metrics_path(p)
        if not os.path.exists(mp):
            print("WARN missing", mp)
            continue
        all_runs[label] = parse_run(mp)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(all_runs, os.path.join(args.out_dir, args.tag + ".csv"))
    make_plot(all_runs, os.path.join(args.out_dir, args.tag + ".png"))

    # concise stdout summary: degraded aligned_abs_rel + span first vs last epoch
    print("\n=== erosion summary (degraded sets) ===")
    for label, sets in all_runs.items():
        for name, rows in sorted(sets.items()):
            if "degraded" not in name or not rows:
                continue
            a, b = rows[0], rows[-1]
            # find peak (min aligned_abs_rel)
            peak = min(rows, key=lambda r: (r.get("aligned_abs_rel") is None,
                                            r.get("aligned_abs_rel") or 1e9))
            print("%-10s %-26s ep%d->ep%d  aligned %.3f->%.3f (peak %.3f@ep%d)  span %.1f->%.1f"
                  % (label, name, a["epoch"], b["epoch"],
                     a.get("aligned_abs_rel") or float("nan"),
                     b.get("aligned_abs_rel") or float("nan"),
                     peak.get("aligned_abs_rel") or float("nan"), peak["epoch"],
                     a.get("pred_depth_span") or float("nan"),
                     b.get("pred_depth_span") or float("nan")))


if __name__ == "__main__":
    main()
