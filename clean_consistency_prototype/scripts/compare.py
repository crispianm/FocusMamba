#!/usr/bin/env python3
"""Cross-arm comparison for the clean<->degraded consistency prototype.

For each arm (run dir with metrics.jsonl) report the degraded-val erosion, span
and boundary retention, the clean-val trajectory (specificity check), and the
consistency-term activity. Writes a summary CSV + trajectory plots.

Usage:
    python compare.py --runs-dir clean_consistency_prototype/runs \
        --arms c0_control si_w0p5 si_w1p0 si_w2p0 logl1_w0p5 logl1_w1p0 logl1_w2p0 \
        --out-dir clean_consistency_prototype/results
"""
import argparse
import json
import os


def load_traj(metrics_path):
    """Return {epoch: {key: val}} for validation events (epoch 0 = pretrain)."""
    rows = {}
    aux = {}
    for line in open(metrics_path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ev = d.get("event")
        if ev in ("validation", "pretrain_validation"):
            e = 0 if ev == "pretrain_validation" else d.get("epoch", 0)
            rows[e] = d.get("metrics", {})
        elif ev == "train_epoch":
            # capture the consistency term if logged at epoch granularity
            m = d.get("metrics", {})
            for k in ("aux/clean_depth_consistency", "train/aux/clean_depth_consistency"):
                if k in m:
                    aux[d.get("epoch")] = m[k]
    return rows, aux


def g(metrics, name, key):
    return metrics.get("%s_%s" % (name, key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="clean_consistency_prototype/runs")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out-dir", default="clean_consistency_prototype/results")
    ap.add_argument("--primary", default="val_vkitti_degraded")
    ap.add_argument("--clean", default="val_vkitti_clean")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    trajs = {}
    for arm in args.arms:
        mp = os.path.join(args.runs_dir, arm, "metrics.jsonl")
        if not os.path.exists(mp):
            print("MISSING", mp)
            continue
        rows, aux = load_traj(mp)
        eps = sorted(e for e in rows if e >= 1)
        if not eps:
            print("no epochs", arm)
            continue
        deg = [(e, g(rows[e], args.primary, "aligned_abs_rel")) for e in eps]
        deg = [(e, v) for e, v in deg if v is not None]
        span = {e: g(rows[e], args.primary, "pred_depth_span") for e in eps}
        bf1 = {e: g(rows[e], args.primary, "boundary_f1") for e in eps}
        cln = [(e, g(rows[e], args.clean, "aligned_abs_rel")) for e in eps]
        cln = [(e, v) for e, v in cln if v is not None]
        best_e, best_v = min(deg, key=lambda x: x[1])
        last_e, last_v = deg[-1]
        trajs[arm] = dict(eps=[e for e, _ in deg], deg=[v for _, v in deg],
                          cln=[v for _, v in cln], span=span, bf1=bf1)
        summary.append(dict(
            arm=arm, best_ep=best_e, best_deg=best_v, last_deg=last_v,
            erosion=last_v - best_v,
            span_best=span.get(best_e), span_last=span.get(last_e),
            bf1_best=bf1.get(best_e), bf1_last=bf1.get(last_e),
            clean_last=cln[-1][1] if cln else None,
        ))

    # control reference
    ctrl = next((s for s in summary if s["arm"] in ("c0_control", args.arms[0])), None)
    cols = ["arm", "best_ep", "best_deg", "last_deg", "erosion", "span_best",
            "span_last", "bf1_best", "bf1_last", "clean_last"]
    out_csv = os.path.join(args.out_dir, "stage1_summary.csv")
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + ",erosion_vs_ctrl_pct\n")
        print("\n%-14s %-7s %-9s %-9s %-9s %-11s %-11s %-9s" % (
            "arm", "best_ep", "best_deg", "last_deg", "erosion", "span b->l", "bf1 b->l", "clean_last"))
        for s in summary:
            er_pct = ""
            if ctrl and ctrl["erosion"]:
                er_pct = "%.0f%%" % (100.0 * s["erosion"] / ctrl["erosion"])
            f.write(",".join("%s" % s.get(c, "") for c in cols) + ",%s\n" % er_pct)
            print("%-14s ep%-5d %-9.4f %-9.4f %-9.4f %5.1f->%-5.1f %0.3f->%-0.3f %-.4f  (er %s ctrl)" % (
                s["arm"], s["best_ep"], s["best_deg"], s["last_deg"], s["erosion"],
                s["span_best"] or -1, s["span_last"] or -1,
                s["bf1_best"] or -1, s["bf1_last"] or -1, s["clean_last"] or -1, er_pct))
    print("\nwrote", out_csv)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable (%s); skipping plots" % e)
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for arm, t in trajs.items():
        ax[0].plot(t["eps"], t["deg"], marker="o", ms=3, label=arm)
        ax[1].plot(t["eps"], [t["span"][e] for e in t["eps"]], marker="o", ms=3, label=arm)
        ax[2].plot(t["eps"], t["cln"], marker="o", ms=3, label=arm)
    ax[0].set_title("VKITTI degraded aligned_abs_rel (erosion)"); ax[0].set_ylabel("aligned_abs_rel")
    ax[1].set_title("VKITTI degraded pred_depth_span (collapse)"); ax[1].set_ylabel("span")
    ax[2].set_title("VKITTI clean aligned_abs_rel (specificity)"); ax[2].set_ylabel("aligned_abs_rel")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "stage1_trajectories.png")
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
