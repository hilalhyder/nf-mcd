"""
Overlap-detection follow-up: does NF-MCD detect overlap when the synthetic
generator gives overlapping nodes REAL signal (blend_strength > 0), instead
of the label-only overlap_rate tested in run_overlap.py (which found ~chance
AUROC because the original generator gives overlapping nodes zero footprint
in edges or content -- see experiments/overlap_summary.log)?

generate_synthetic_multimodal_graph gained a new `blend_strength` parameter
(nf_mcd/datasets.py), default 0.0, verified to reproduce the original
generator's output byte-for-byte for any seed (demo.py's modularity /
overlapping_nmi / membership_f1 / rule_fidelity match exactly before and
after the edit -- 0.470 / 0.729 / 0.927 / 0.333). This script sweeps
blend_strength > 0 and measures whether NF-MCD's overlap detection actually
improves once there is real signal to find.

Usage (from nfmcd_impl/):
    py run_overlap2.py run [--workers N]
    py run_overlap2.py summary

Crash-safe: each finished job appended to experiments/overlap2_results.csv
and experiments/overlap2_thresholds.csv (flush+fsync); reruns skip finished
jobs; other files written temp+rename.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "overlap2_results.csv")
CURVE_CSV = os.path.join(EXP, "overlap2_thresholds.csv")
SUMMARY_LOG = os.path.join(EXP, "overlap2_summary.log")
PROGRESS_MD = os.path.join(EXP, "overlap2_progress.md")
AUROC_PNG = os.path.join(EXP, "overlap2_auroc.png")
PRIMARY_PNG = os.path.join(EXP, "overlap2_primary_cost.png")

from run_overlap import (  # noqa: E402
    binary_prf, overlap_curve_and_summary, second_community_accuracy,
    synthetic_primary, atomic_write_text, append_row, load_done,
)

RATES = (0.1, 0.2, 0.3, 0.4)
BLENDS = (0.0, 0.3, 0.5, 0.7, 1.0)
SEEDS = (0, 1, 2, 3, 4)

RESULT_FIELDS = [
    "rate", "blend", "method", "seed", "n_nodes", "true_overlap_rate", "n_true_overlap",
    "auroc", "ap", "best_thr", "best_f1", "precision_at_best", "recall_at_best",
    "f1_at_0.2", "precision_at_0.2", "recall_at_0.2",
    "second_comm_acc", "n_eval_second",
    "primary_onmi", "primary_modularity", "primary_f1",
    "note", "error", "secs",
]
CURVE_FIELDS = ["rate", "blend", "method", "seed", "threshold", "precision", "recall", "f1"]


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Overlap follow-up (blend_strength) progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_overlap2.py run --workers 4\n"
        "py run_overlap2.py summary\n"
        "```\n"
    ))


def job(rate, blend, seed):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    from nf_mcd.pipeline import NFMCD
    from nf_mcd import baselines as blib

    out = []
    data = generate_synthetic_multimodal_graph(
        n_nodes=300, n_communities=5, overlap_rate=rate, seed=1000 + seed, blend_strength=blend,
    )
    n = data.G.number_of_nodes()
    true_bin = np.array([len(t) > 1 for t in data.true_communities])
    true_primary = synthetic_primary(n, data.n_communities)
    base = dict(rate=rate, blend=blend, seed=seed, n_nodes=n,
                true_overlap_rate=round(float(true_bin.mean()), 4), n_true_overlap=int(true_bin.sum()))

    for method, ctor in (
        ("nfmcd_default", lambda: NFMCD(n_communities=data.n_communities, seed=seed)),
        ("nfmcd_robust", lambda: NFMCD.robust(n_communities=data.n_communities, seed=seed)),
    ):
        t0 = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ctor().fit(data.G, text_embeddings=data.text_embeddings, image_embeddings=data.image_embeddings)
            ev = model.evaluate(true_communities_per_node=data.true_communities)
        U, hard = model.U_, model.predict_hard()
        curve, summ = overlap_curve_and_summary(U, true_bin)
        acc, n_eval = second_community_accuracy(U, hard, true_primary, data.true_communities, data.n_communities)
        row = dict(base, method=method, second_comm_acc=round(acc, 4) if n_eval else "",
                   n_eval_second=n_eval,
                   primary_onmi=round(ev["overlapping_nmi"], 4),
                   primary_modularity=round(ev["modularity"], 4),
                   primary_f1=round(ev["membership_f1"], 4),
                   note="", error="", secs=round(time.monotonic() - t0, 2), **summ)
        out.append((row, [dict(rate=rate, blend=blend, seed=seed, method=method,
                                threshold=t, precision=p, recall=r, f1=f) for t, p, r, f in curve]))

    d_min = dict(G=data.G, true=data.true_communities, n_communities=data.n_communities)
    for method, fn in (("demon", blib.demon), ("slpa", blib.slpa)):
        t0 = time.monotonic()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fn(d_min, seed)
            coverage = np.zeros(n, dtype=int)
            for comm in res.view:
                for i in comm:
                    coverage[i] += 1
            pred_bin = coverage > 1
            p, r, f1 = binary_prf(true_bin, pred_bin)
            row = dict(base, method=method, auroc="", ap="", best_thr="", best_f1=round(f1, 4),
                       precision_at_best=round(p, 4), recall_at_best=round(r, 4),
                       **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                       second_comm_acc="", n_eval_second="",
                       primary_onmi="", primary_modularity="", primary_f1="",
                       note="single fixed operating point", error="",
                       secs=round(time.monotonic() - t0, 2))
        except Exception as exc:  # noqa: BLE001
            row = dict(base, method=method, auroc="", ap="", best_thr="", best_f1="",
                       precision_at_best="", recall_at_best="",
                       **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                       second_comm_acc="", n_eval_second="",
                       primary_onmi="", primary_modularity="", primary_f1="",
                       note="", error=f"{type(exc).__name__}: {exc}".replace("\n", " "),
                       secs=round(time.monotonic() - t0, 2))
        out.append((row, []))

    p_always, r_always, f1_always = binary_prf(true_bin, np.ones(n, dtype=bool))
    p_never, r_never, f1_never = binary_prf(true_bin, np.zeros(n, dtype=bool))
    for method, (p, r, f1) in (
        ("always_overlap", (p_always, r_always, f1_always)),
        ("always_no_overlap", (p_never, r_never, f1_never)),
    ):
        out.append((dict(base, method=method, auroc="", ap="", best_thr="", best_f1=round(f1, 4),
                          precision_at_best=round(p, 4), recall_at_best=round(r, 4),
                          **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                          second_comm_acc="", n_eval_second="",
                          primary_onmi="", primary_modularity="", primary_f1="",
                          note="single fixed operating point", error="", secs=0.0), []))
    return out


def all_jobs():
    return [(r, b, s) for r in RATES for b in BLENDS for s in SEEDS]


def run_job(j):
    rate, blend, seed = j
    try:
        return rate, blend, seed, job(rate, blend, seed), None
    except Exception as exc:  # noqa: BLE001
        return rate, blend, seed, None, f"{type(exc).__name__}: {exc}".replace("\n", " ")


def _job_key(rate, blend, seed):
    return (str(rate), str(blend), str(seed))


def cmd_run(workers):
    jobs = all_jobs()
    done_keys = load_done(RESULTS_CSV, ["rate", "blend", "seed"]) if os.path.exists(RESULTS_CSV) else set()
    pending = [j for j in jobs if _job_key(*j) not in done_keys]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; running {len(pending)} with {workers} workers")
    write_progress("run", len(jobs) - len(pending), len(jobs), "starting")
    n_done = len(jobs) - len(pending)
    if not pending:
        write_progress("run", n_done, len(jobs), "all done")
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in pending}
        for fut in as_completed(futs):
            rate, blend, seed, rows, err = fut.result()
            if err:
                print(f"  FAILED rate={rate} blend={blend} seed={seed}: {err}")
                append_row(RESULTS_CSV, RESULT_FIELDS, dict(
                    rate=rate, blend=blend, method="ALL", seed=seed, n_nodes="", true_overlap_rate="",
                    n_true_overlap="", auroc="", ap="", best_thr="", best_f1="", precision_at_best="",
                    recall_at_best="", **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                    second_comm_acc="", n_eval_second="",
                    primary_onmi="", primary_modularity="", primary_f1="",
                    note="", error=err, secs=0))
            else:
                for row, curve_rows in rows:
                    append_row(RESULTS_CSV, RESULT_FIELDS, row)
                    for cr in curve_rows:
                        append_row(CURVE_CSV, CURVE_FIELDS, cr)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(jobs):
                write_progress("run", n_done, len(jobs), f"last: rate={rate} blend={blend} seed={seed}")
    write_progress("run", len(jobs), len(jobs), "all done")
    print("done")


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def group_mean(rows_subset, field):
    vals = [_fnum(r[field]) for r in rows_subset if r[field] not in ("", None)]
    vals = [v for v in vals if not np.isnan(v)]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)


def cmd_summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(RESULTS_CSV)
    lines = []

    lines.append("Overlap follow-up: does giving overlap REAL signal (blend_strength) change detection?")
    lines.append("n_nodes=300, k=5, 5 seeds. blend_strength=0.0 reproduces run_overlap.py's Part A")
    lines.append("(sanity check at the bottom).")
    lines.append("=" * 100)

    for rate in RATES:
        lines.append(f"\n\noverlap_rate={rate}")
        lines.append("-" * 100)
        for blend in BLENDS:
            sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend]
            if not sub:
                continue
            true_rate = group_mean(sub, "true_overlap_rate")[0]
            lines.append(f"\n  blend_strength={blend}  (observed true overlap rate: {true_rate:.3f})")
            lines.append(f"  {'method':<20}{'AUROC':>8}{'AP':>8}{'best_thr':>9}{'best_F1':>9}"
                          f"{'F1@0.2':>8}{'2nd-acc':>9}{'primary_ONMI':>14}")
            for m in ("nfmcd_default", "nfmcd_robust", "demon", "slpa", "always_overlap", "always_no_overlap"):
                ms = [r for r in sub if r["method"] == m]
                if not ms:
                    continue
                auroc = group_mean(ms, "auroc")[0]
                ap = group_mean(ms, "ap")[0]
                bthr = group_mean(ms, "best_thr")[0]
                bf1 = group_mean(ms, "best_f1")[0]
                f102 = group_mean(ms, "f1_at_0.2")[0]
                acc = group_mean(ms, "second_comm_acc")[0]
                ponmi = group_mean(ms, "primary_onmi")[0]

                def fmt(x):
                    return f"{x:.3f}" if not np.isnan(x) else "-"
                lines.append(f"  {m:<20}{fmt(auroc):>8}{fmt(ap):>8}{fmt(bthr):>9}{fmt(bf1):>9}"
                              f"{fmt(f102):>8}{fmt(acc):>9}{fmt(ponmi):>14}")

    lines.append("\n\nPrimary-assignment cost of injecting real overlap (nfmcd_default), ONMI vs blend_strength")
    lines.append("=" * 100)
    for rate in RATES:
        lines.append(f"\noverlap_rate={rate}")
        for blend in BLENDS:
            sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                   and r["method"] == "nfmcd_default"]
            if not sub:
                continue
            ponmi_mean, ponmi_sd, _ = group_mean(sub, "primary_onmi")
            pf1_mean, _, _ = group_mean(sub, "primary_f1")
            lines.append(f"  blend={blend}: primary_onmi={ponmi_mean:.3f} (sd {ponmi_sd:.3f})  "
                          f"primary_f1={pf1_mean:.3f}")

    lines.append("\n\nSanity check: blend_strength=0.0 here vs run_overlap.py's original Part A (should match)")
    lines.append("=" * 100)
    orig_path = os.path.join(EXP, "overlap_results.csv")
    if os.path.exists(orig_path):
        orig_rows = _read_csv(orig_path)
        for rate in RATES:
            here = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == 0.0
                    and r["method"] == "nfmcd_default"]
            orig = [r for r in orig_rows if r["part"] == "A" and r["dataset"] == f"r{rate}"
                    and r["method"] == "nfmcd_default"]
            if here and orig:
                h_auroc = group_mean(here, "auroc")[0]
                o_auroc = group_mean(orig, "auroc")[0]
                lines.append(f"  rate={rate}: here AUROC={h_auroc:.3f}  original run_overlap.py AUROC={o_auroc:.3f}")

    atomic_write_text(SUMMARY_LOG, "\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_LOG}")

    try:
        fig, axes = plt.subplots(1, len(RATES), figsize=(4.2 * len(RATES), 3.8), sharey=True)
        for ax, rate in zip(axes, RATES):
            for method, style in (("nfmcd_default", "-o"), ("nfmcd_robust", "-s"),
                                   ("demon", "--^"), ("slpa", "--v")):
                xs, ys = [], []
                for blend in BLENDS:
                    sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                           and r["method"] == method]
                    if method in ("demon", "slpa"):
                        v = group_mean(sub, "best_f1")[0]
                    else:
                        v = group_mean(sub, "auroc")[0]
                    if not np.isnan(v):
                        xs.append(blend)
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, style, label=method, markersize=5)
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"overlap_rate={rate}")
            ax.set_xlabel("blend_strength")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("AUROC (NF-MCD) / F1 (DEMON, SLPA)")
        axes[0].legend(fontsize=7)
        fig.suptitle("Overlap detection vs. blend_strength (0.5 dotted = chance AUROC)")
        fig.tight_layout()
        fig.savefig(AUROC_PNG, dpi=130)
        plt.close(fig)
        print(f"wrote {AUROC_PNG}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot 1 failed: {exc}")

    try:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for rate in RATES:
            xs, ys = [], []
            for blend in BLENDS:
                sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                       and r["method"] == "nfmcd_default"]
                v = group_mean(sub, "primary_onmi")[0]
                if not np.isnan(v):
                    xs.append(blend)
                    ys.append(v)
            ax.plot(xs, ys, "-o", label=f"overlap_rate={rate}", markersize=5)
        ax.set_xlabel("blend_strength")
        ax.set_ylabel("primary overlapping_nmi (nfmcd_default)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Primary-assignment cost of adding real overlap signal")
        fig.tight_layout()
        fig.savefig(PRIMARY_PNG, dpi=130)
        plt.close(fig)
        print(f"wrote {PRIMARY_PNG}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot 2 failed: {exc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if args.cmd == "run":
        cmd_run(args.workers)
    else:
        cmd_summary()
