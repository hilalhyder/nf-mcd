"""
Overlap-threshold sensitivity sweep (reviewer request, paper Sections 6.1-6.3):
the main evaluation thresholds fuzzy memberships U at 0.2 to get an overlapping
hard-style view for ONMI/F1. This fits NFMCD once per (dataset, variant, seed)
-- same protocol as run_baselines.py/run_explain.py -- and re-thresholds that
SAME fitted U at several thresholds, so the fit cost is paid once per run, not
once per threshold.

Usage (from nfmcd_impl/):
    py run_threshold_sweep.py
Writes experiments/threshold_sweep_results.csv and
experiments/threshold_sweep_summary.log.
"""
from __future__ import annotations

import csv
import os
import pickle
import warnings

import numpy as np

from nf_mcd import baselines as b
from nf_mcd import metrics as mx
from nf_mcd.pipeline import NFMCD

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "threshold_sweep_results.csv")
SUMMARY_LOG = os.path.join(EXP, "threshold_sweep_summary.log")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon", "fakeddit_real_lcc"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
VARIANTS = ["default", "robust"]
SEEDS = [0, 1, 2]


def fit_model(d, seed, variant):
    n = d["n_communities"]
    model = NFMCD(n_communities=n, seed=seed) if variant == "default" else NFMCD.robust(n_communities=n, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return model


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    rows = []

    for ds in DATASETS:
        cache_path = os.path.join(CACHE, f"{ds}.pkl")
        if not os.path.exists(cache_path):
            print(f"  skip {ds}: no dataset cache found")
            continue
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        true_view = b._true_view(d["true"], d["n_communities"])
        n = d["G"].number_of_nodes()

        for variant in VARIANTS:
            models = {}
            for seed in SEEDS:
                models[seed] = fit_model(d, seed, variant)
            for t in THRESHOLDS:
                for seed, model in models.items():
                    view = model.overlapping_communities_view(threshold=t)
                    onmi = float(mx.overlapping_nmi(view, true_view, n))
                    f1 = float(mx.membership_f1(view, true_view))
                    max_u = float(model.U_.max(axis=1).mean())
                    rows.append(dict(dataset=ds, variant=variant, threshold=t, seed=seed,
                                      onmi=onmi, f1=f1, mean_max_membership=max_u))
                    print(f"  {ds:<20}{variant:<9}t={t:.2f} seed={seed}  onmi={onmi:.4f} f1={f1:.4f}")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "threshold", "seed", "onmi", "f1", "mean_max_membership"])
        w.writeheader()
        w.writerows(rows)

    # Summary: mean±sd over seeds, per dataset/variant, ONMI and F1 across thresholds.
    lines = []
    lines.append("Overlap-threshold sensitivity sweep (Sections 6.1-6.3)")
    lines.append("Re-scores cached fitted models at each threshold; no refitting.")
    lines.append(f"Thresholds tested: {THRESHOLDS}")
    lines.append("")

    for variant in VARIANTS:
        lines.append("=" * 100)
        lines.append(f"VARIANT: {variant}")
        lines.append("=" * 100)
        for metric in ("onmi", "f1"):
            lines.append(f"\n{metric.upper()} by threshold (mean±sd over seeds)")
            head = f"{'threshold':<10}" + "".join(f"{ds:>20}" for ds in DATASETS)
            lines.append(head)
            for t in THRESHOLDS:
                row = f"{t:<10.2f}"
                for ds in DATASETS:
                    vals = [r[metric] for r in rows if r["dataset"] == ds and r["variant"] == variant and r["threshold"] == t]
                    if vals:
                        a = np.array(vals)
                        row += f"{a.mean():>13.3f}±{a.std(ddof=0):<5.3f}"
                    else:
                        row += f"{'n/a':>20}"
                lines.append(row)
        lines.append(f"\nMean max membership (soft-ness check, independent of threshold) by dataset:")
        for ds in DATASETS:
            vals = [r["mean_max_membership"] for r in rows if r["dataset"] == ds and r["variant"] == variant and r["threshold"] == THRESHOLDS[0]]
            if vals:
                lines.append(f"  {ds:<20}{np.mean(vals):.3f}")
        lines.append("")

    # Best threshold per dataset/variant by ONMI, and delta vs the 0.2 default.
    lines.append("=" * 100)
    lines.append("Best threshold by ONMI, and cost of the fixed 0.2 default relative to it")
    lines.append("=" * 100)
    for variant in VARIANTS:
        for ds in DATASETS:
            by_t = {}
            for t in THRESHOLDS:
                vals = [r["onmi"] for r in rows if r["dataset"] == ds and r["variant"] == variant and r["threshold"] == t]
                if vals:
                    by_t[t] = float(np.mean(vals))
            if not by_t:
                continue
            best_t = max(by_t, key=by_t.get)
            at_02 = by_t.get(0.20, float("nan"))
            cost = by_t[best_t] - at_02
            lines.append(f"  {ds:<20}{variant:<9}best={best_t:.2f} (onmi={by_t[best_t]:.3f})  "
                         f"at 0.20 (onmi={at_02:.3f})  cost_of_fixed_0.2={cost:+.3f}")

    text = "\n".join(lines) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print("\n" + text)
    print(f"Wrote {RESULTS_CSV} and {SUMMARY_LOG}")


if __name__ == "__main__":
    main()
