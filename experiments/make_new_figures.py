# -*- coding: utf-8 -*-
"""Polished replacements/additions for the manuscript: factor-sweep surface,
calibration reliability diagrams, and synthetic-vs-real structural comparison.
Reads the raw CSVs the corresponding run_*.py scripts already produced."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})

METHOD_LABELS = {
    "structure_only": "Structure only",
    "content_only": "Content only",
    "simple_concat": "Simple concat.",
    "spectral_content": "Spectral+content",
    "nfmcd_alpha_fixed": "NF-MCD (fixed-α)",
    "nfmcd_adaptive": "NF-MCD (adaptive-α)",
}
METHOD_ORDER = list(METHOD_LABELS.keys())
REGIME_LABELS = {
    "strong_struct_weak_content": "Strong structure / weak content",
    "weak_struct_strong_content": "Weak structure / strong content",
}


def fig7_factor_sweep():
    df = pd.read_csv(os.path.join(OUT, "factor_sweep_results.csv"))
    df = df[df.error.isna()]
    regimes = list(REGIME_LABELS.keys())
    fig, axes = plt.subplots(2, 6, figsize=(16.5, 6.2), constrained_layout=True)
    im = None
    for ri, regime in enumerate(regimes):
        for mi, method in enumerate(METHOD_ORDER):
            ax = axes[ri, mi]
            sub = df[(df.regime == regime) & (df.method == method)]
            piv = sub.pivot_table(index="agreement", columns="missingness", values="onmi", aggfunc="mean")
            piv = piv.sort_index(ascending=False)
            im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"{c:.1f}" for c in piv.columns], fontsize=8, rotation=0)
            if mi == 0:
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels([f"{r:.1f}" for r in piv.index], fontsize=8)
                ax.set_ylabel(REGIME_LABELS[regime] + "\ncross-modal agreement", fontsize=9)
            else:
                ax.set_yticks([])
            if ri == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=10.5)
            if ri == 1:
                ax.set_xlabel("missingness", fontsize=8.5)
            ax.grid(False)
    cax = fig.add_axes((1.005, 0.15, 0.012, 0.7))
    fig.colorbar(im, cax=cax, label="Mean overlapping NMI (5 seeds)")
    fig.suptitle("Controlled factor sweep: ONMI vs. cross-modal agreement × missingness (n=320, k=4)",
                 fontsize=13, y=1.04)
    fig.savefig(os.path.join(OUT, "fig7_factor_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_factor_sweep.png")


def fig8_calibration():
    df = pd.read_csv(os.path.join(OUT, "calibration_results.csv"))
    df = df[(df.kind == "default") & (df.error.isna())]
    panels = [
        ("synthetic", "hi", 0.2, 0.0, "Synthetic (strong content), 0% missing"),
        ("synthetic", "hi", 0.2, 0.4, "Synthetic (strong content), 40% missing"),
        ("real", "crisismmd", 0.2, 0.0, "CrisisMMD (real), 0% missing"),
        ("real", "crisismmd", 0.2, 0.4, "CrisisMMD (real), 40% missing"),
        ("real", "fakeddit", 0.2, 0.0, "Fakeddit (real), 0% missing"),
        ("real", "fakeddit", 0.2, 0.4, "Fakeddit (real), 40% missing"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    color = "#2a78d6"
    for ax, (source, regime, param, missingness, title) in zip(axes.flat, panels):
        sub = df[(df.source == source) & (df.regime == regime) & (df.param == param) & (df.missingness == missingness)]
        agg = {}
        for _, row in sub.iterrows():
            bins = json.loads(row.bins_json)
            for b in bins:
                if b["n"] == 0 or b["conf"] is None:
                    continue
                key = (b["lo"], b["hi"])
                agg.setdefault(key, {"n": 0, "conf_sum": 0.0, "acc_sum": 0.0})
                agg[key]["n"] += b["n"]
                agg[key]["conf_sum"] += b["conf"] * b["n"]
                agg[key]["acc_sum"] += b["acc"] * b["n"]
        keys = sorted(agg.keys())
        confs = [agg[k]["conf_sum"] / agg[k]["n"] for k in keys]
        accs = [agg[k]["acc_sum"] / agg[k]["n"] for k in keys]
        ns = [agg[k]["n"] for k in keys]
        ax.plot([0, 1], [0, 1], ls=":", color="#8a8985", lw=1, zorder=1)
        ax.plot(confs, accs, marker="o", color=color, ms=6, lw=1.8, zorder=3)
        for c, a, n in zip(confs, accs, ns):
            ax.annotate(str(n), (c, a), textcoords="offset points", xytext=(6, -3), fontsize=7, color="#52514e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("mean predicted P(mismatch)", fontsize=9)
        ax.set_ylabel("empirical mismatch rate", fontsize=9)
    fig.suptitle("Reliability diagrams: NF-MCD confidence as a mismatch-probability estimator\n"
                 "(default rank cap, pooled over seeds; point labels = node count in bin)",
                 fontsize=12.5, y=1.05)
    fig.savefig(os.path.join(OUT, "fig8_calibration.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_calibration.png")


def fig9_synth_vs_real():
    df = pd.read_csv(os.path.join(OUT, "synth_vs_real_results.csv"))
    df = df.set_index("dataset")
    real = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
    synth = ["synthetic_typical", "synthetic_stress20k"]
    order = real + synth
    labels = {"crisismmd": "CrisisMMD", "pheme": "PHEME", "fakeddit": "Fakeddit", "dblp": "DBLP",
              "amazon": "Amazon", "synthetic_typical": "Synthetic\n(typical)",
              "synthetic_stress20k": "Synthetic\n(20k stress)"}
    dims = [
        ("deg_mean", "Mean degree"),
        ("avg_clustering", "Clustering coefficient"),
        ("comm_size_mean", "Mean community size"),
        ("modularity_gt_primary", "Modularity (ground truth)"),
        ("overlap_frac", "Overlap fraction"),
        ("content_delta", "Content-similarity, within−between"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    colors = ["#2a78d6"] * len(real) + ["#eb6834"] * len(synth)
    for ax, (col, title) in zip(axes.flat, dims):
        vals = [df.loc[d, col] if d in df.index and pd.notna(df.loc[d, col]) else np.nan for d in order]
        xs = np.arange(len(order))
        bars = ax.bar(xs, vals, color=colors, width=0.65)
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[d] for d in order], fontsize=8, rotation=35, ha="right")
        ax.set_title(title, fontsize=10.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2g}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2a78d6"), plt.Rectangle((0, 0), 1, 1, color="#eb6834")]
    fig.legend(handles, ["Real dataset", "Synthetic generator"], loc="upper center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Synthetic generator vs. real datasets, six structural dimensions", fontsize=13, y=1.12)
    fig.savefig(os.path.join(OUT, "fig9_synth_vs_real.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig9_synth_vs_real.png")


if __name__ == "__main__":
    fig7_factor_sweep()
    fig8_calibration()
    fig9_synth_vs_real()
