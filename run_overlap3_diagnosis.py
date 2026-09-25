"""
Diagnose why giving synthetic "overlapping" nodes real signal (blend_strength,
added to nf_mcd.datasets.generate_synthetic_multimodal_graph in the last run)
caused primary-assignment ONMI to mildly RISE with blend_strength instead of
falling (run_overlap2.py: overlap_rate=0.4, ONMI 0.644 at blend=0 -> 0.711 at
blend=1.0). Diagnosis only -- reads nf_mcd/*, does not modify it.

Four candidate mechanisms, each measured directly (not narrated):
  1. degree artifact       -- overlapping nodes gain edges as blend_strength
                               rises; does raw degree correlate with getting
                               the (strict) primary assignment right?
  2. scoring/credit artifact -- the "primary_onmi" reported in run_overlap2.py
                               is NFMCD.evaluate()'s overlapping_nmi, scored
                               against the FULL true_communities set (which
                               includes both c1 and c2 for overlapping nodes)
                               and against the model's full >=0.2 membership
                               view (not just argmax). A node whose hard
                               assignment lands on c2 (not the nominal primary
                               c1) still credits as "in community c2" in both
                               the true view and the predicted view. Test:
                               strict primary accuracy (must map to c1) vs.
                               lenient accuracy (may map to c1 OR c2) using
                               the node's HARD (argmax) assignment, Hungarian-
                               aligned to true community ids.
  3. structural spillover  -- restrict scoring to NON-overlapping nodes only
                               (no scoring ambiguity there); does their
                               accuracy also rise with blend_strength? If so
                               the effect is graph-wide, not just node-local.
  4. edge-vs-content decomposition -- generate_synthetic_multimodal_graph's
                               rng draw sequence for text/image noise is
                               IDENTICAL regardless of blend_strength (only
                               the centroid mix changes: same number/order of
                               rng.normal/rng.random calls); blend_strength's
                               extra edges are drawn from an independent RNG
                               stream. So calling the generator twice at the
                               same seed (blend_strength=B and 0.0) and
                               swapping G <-> content between the two calls
                               gives clean edge-only / content-only ablations
                               without touching library code.

Usage (from nfmcd_impl/):
    py run_overlap3_diagnosis.py run [--workers N]
    py run_overlap3_diagnosis.py summary
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
RESULTS_CSV = os.path.join(EXP, "overlap3_results.csv")
SUMMARY_LOG = os.path.join(EXP, "overlap3_summary.log")
PROGRESS_MD = os.path.join(EXP, "overlap3_progress.md")
DIAG_PNG = os.path.join(EXP, "overlap3_diagnosis.png")

from run_overlap import synthetic_primary, hungarian_align, atomic_write_text, append_row, load_done  # noqa: E402

RATES = (0.1, 0.4)
BLENDS = (0.0, 0.3, 0.5, 0.7, 1.0)
SEEDS = (0, 1, 2, 3, 4)
CONDITIONS_NONZERO = ("combined", "edge_only", "content_only")

RESULT_FIELDS = [
    "rate", "blend", "condition", "seed", "n_nodes", "n_overlap",
    "onmi_standard",
    "strict_acc_all", "lenient_acc_all",
    "strict_acc_overlap", "lenient_acc_overlap",
    "strict_acc_nonoverlap",
    "frac_hard_c1", "frac_hard_c2", "frac_hard_neither",
    "mean_degree_overlap", "mean_degree_nonoverlap",
    "spearman_degree_vs_strict_correct",
    "note", "error", "secs",
]


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Overlap primary-accuracy-rise diagnosis progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_overlap3_diagnosis.py run --workers 4\n"
        "py run_overlap3_diagnosis.py summary\n"
        "```\n"
    ))


def _fit_and_score(G, text_embeddings, image_embeddings, true_communities, n_communities,
                    true_primary, seed):
    from nf_mcd.pipeline import NFMCD
    from scipy.stats import spearmanr

    n = G.number_of_nodes()
    overlap_mask = np.array([len(t) > 1 for t in true_communities])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NFMCD(n_communities=n_communities, seed=seed)
        model.fit(G, text_embeddings=text_embeddings, image_embeddings=image_embeddings)
        ev = model.evaluate(true_communities_per_node=true_communities)

    hard = model.predict_hard()
    mapping = hungarian_align(hard, true_primary, n_communities)
    pred_mapped = np.array([mapping.get(int(h), -1) for h in hard])

    strict_correct = pred_mapped == true_primary
    lenient_correct = np.array([pred_mapped[i] in true_communities[i] for i in range(n)])

    frac_c1 = frac_c2 = frac_neither = float("nan")
    if overlap_mask.sum() > 0:
        c1s = true_primary[overlap_mask]
        c2s = np.array([next(c for c in true_communities[i] if c != true_primary[i])
                         for i in range(n) if overlap_mask[i]])
        pm = pred_mapped[overlap_mask]
        frac_c1 = float(np.mean(pm == c1s))
        frac_c2 = float(np.mean(pm == c2s))
        frac_neither = float(np.mean((pm != c1s) & (pm != c2s)))

    deg = np.array([G.degree(v) for v in G.nodes()])
    mean_deg_ov = float(deg[overlap_mask].mean()) if overlap_mask.sum() else float("nan")
    mean_deg_non = float(deg[~overlap_mask].mean()) if (~overlap_mask).sum() else float("nan")

    rho = float("nan")
    if len(set(deg.tolist())) > 1 and 0 < strict_correct.sum() < n:
        rho = float(spearmanr(deg, strict_correct.astype(int)).correlation)

    return dict(
        n_nodes=n, n_overlap=int(overlap_mask.sum()),
        onmi_standard=round(ev["overlapping_nmi"], 4),
        strict_acc_all=round(float(strict_correct.mean()), 4),
        lenient_acc_all=round(float(lenient_correct.mean()), 4),
        strict_acc_overlap=round(float(strict_correct[overlap_mask].mean()), 4) if overlap_mask.sum() else "",
        lenient_acc_overlap=round(float(lenient_correct[overlap_mask].mean()), 4) if overlap_mask.sum() else "",
        strict_acc_nonoverlap=round(float(strict_correct[~overlap_mask].mean()), 4) if (~overlap_mask).sum() else "",
        frac_hard_c1=round(frac_c1, 4) if frac_c1 == frac_c1 else "",
        frac_hard_c2=round(frac_c2, 4) if frac_c2 == frac_c2 else "",
        frac_hard_neither=round(frac_neither, 4) if frac_neither == frac_neither else "",
        mean_degree_overlap=round(mean_deg_ov, 3) if mean_deg_ov == mean_deg_ov else "",
        mean_degree_nonoverlap=round(mean_deg_non, 3) if mean_deg_non == mean_deg_non else "",
        spearman_degree_vs_strict_correct=round(rho, 4) if rho == rho else "",
    )


def job(rate, blend, seed):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph

    out = []
    data_base = generate_synthetic_multimodal_graph(
        n_nodes=300, n_communities=5, overlap_rate=rate, seed=1000 + seed, blend_strength=0.0,
    )
    n = data_base.G.number_of_nodes()
    true_primary = synthetic_primary(n, data_base.n_communities)
    base_kwargs = dict(true_communities=data_base.true_communities, n_communities=data_base.n_communities,
                        true_primary=true_primary, seed=seed)

    if blend == 0.0:
        conditions = [("combined", data_base.G, data_base.text_embeddings, data_base.image_embeddings)]
    else:
        data_full = generate_synthetic_multimodal_graph(
            n_nodes=300, n_communities=5, overlap_rate=rate, seed=1000 + seed, blend_strength=blend,
        )
        # true_communities/true_primary are identical between data_base and data_full
        # (same seed -> identical overlap_rate draws, which happen before blend_strength
        # is consulted at all) -- verified by construction of the generator.
        conditions = [
            ("combined", data_full.G, data_full.text_embeddings, data_full.image_embeddings),
            ("edge_only", data_full.G, data_base.text_embeddings, data_base.image_embeddings),
            ("content_only", data_base.G, data_full.text_embeddings, data_full.image_embeddings),
        ]

    for cond_name, G, e_t, e_v in conditions:
        t0 = time.monotonic()
        try:
            scored = _fit_and_score(G, e_t, e_v, seed=seed, **{k: v for k, v in base_kwargs.items() if k != "seed"})
            row = dict(rate=rate, blend=blend, condition=cond_name, seed=seed, note="", error="",
                       secs=round(time.monotonic() - t0, 2), **scored)
        except Exception as exc:  # noqa: BLE001
            row = dict(rate=rate, blend=blend, condition=cond_name, seed=seed,
                       n_nodes="", n_overlap="", onmi_standard="",
                       strict_acc_all="", lenient_acc_all="",
                       strict_acc_overlap="", lenient_acc_overlap="", strict_acc_nonoverlap="",
                       frac_hard_c1="", frac_hard_c2="", frac_hard_neither="",
                       mean_degree_overlap="", mean_degree_nonoverlap="",
                       spearman_degree_vs_strict_correct="",
                       note="", error=f"{type(exc).__name__}: {exc}".replace("\n", " "),
                       secs=round(time.monotonic() - t0, 2))
        out.append(row)
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
    # done_keys from load_done is per-ROW, not per-job (a job writes multiple rows,
    # one per condition, all sharing the same rate/blend/seed) -- rebuild a proper
    # per-job "done" set by requiring the expected row count for that blend.
    import collections
    counts = collections.Counter(done_keys)
    pending = []
    for j in jobs:
        rate, blend, seed = j
        expected = 1 if blend == 0.0 else 3
        if counts[_job_key(rate, blend, seed)] < expected:
            pending.append(j)
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; running {len(pending)} with {workers} workers")
    n_done = len(jobs) - len(pending)
    write_progress("run", n_done, len(jobs), "starting")
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
                    rate=rate, blend=blend, condition="ALL", seed=seed, n_nodes="", n_overlap="",
                    onmi_standard="", strict_acc_all="", lenient_acc_all="",
                    strict_acc_overlap="", lenient_acc_overlap="", strict_acc_nonoverlap="",
                    frac_hard_c1="", frac_hard_c2="", frac_hard_neither="",
                    mean_degree_overlap="", mean_degree_nonoverlap="",
                    spearman_degree_vs_strict_correct="", note="", error=err, secs=0))
            else:
                for row in rows:
                    append_row(RESULTS_CSV, RESULT_FIELDS, row)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(jobs):
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
    lines.append("Diagnosis: why does primary ONMI RISE with blend_strength (real overlap signal)?")
    lines.append("n_nodes=300, k=5, 5 seeds, NF-MCD default. Conditions: combined (edges+content blended,")
    lines.append("matches run_overlap2.py), edge_only, content_only (ablation; blend=0 has only 'combined',")
    lines.append("identical to both ablations trivially).")
    lines.append("=" * 100)

    lines.append("\n\n### Mechanism 2 (scoring/credit) + Mechanism 3 (structural spillover): 'combined' condition")
    lines.append("strict_acc = hard assignment must map to the NOMINAL primary community c1.")
    lines.append("lenient_acc = hard assignment may map to EITHER true community (c1 or c2) -- this is what")
    lines.append("the standard overlapping_nmi metric effectively rewards for overlapping nodes.")
    lines.append("=" * 100)
    for rate in RATES:
        lines.append(f"\noverlap_rate={rate}")
        lines.append(f"  {'blend':<7}{'onmi_std':>10}{'strict_all':>12}{'lenient_all':>13}"
                      f"{'strict_ov':>11}{'lenient_ov':>12}{'strict_nonov':>14}"
                      f"{'hard->c1':>10}{'hard->c2':>10}{'hard->nei':>11}")
        for blend in BLENDS:
            sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                   and r["condition"] == "combined" and not r["error"]]
            if not sub:
                continue

            def g(f):
                return group_mean(sub, f)[0]
            lines.append(f"  {blend:<7}{g('onmi_standard'):>10.3f}{g('strict_acc_all'):>12.3f}"
                          f"{g('lenient_acc_all'):>13.3f}{g('strict_acc_overlap'):>11.3f}"
                          f"{g('lenient_acc_overlap'):>12.3f}{g('strict_acc_nonoverlap'):>14.3f}"
                          f"{g('frac_hard_c1'):>10.3f}{g('frac_hard_c2'):>10.3f}{g('frac_hard_neither'):>11.3f}")

    lines.append("\n\n### Mechanism 1: degree artifact")
    lines.append("Mean degree by node group, and Spearman(degree, strict-correct) pooled over all nodes.")
    lines.append("=" * 100)
    for rate in RATES:
        lines.append(f"\noverlap_rate={rate}")
        lines.append(f"  {'blend':<7}{'deg_overlap':>13}{'deg_nonoverlap':>16}{'spearman(deg,strict_ok)':>26}")
        for blend in BLENDS:
            sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                   and r["condition"] == "combined" and not r["error"]]
            if not sub:
                continue
            do = group_mean(sub, "mean_degree_overlap")[0]
            dn = group_mean(sub, "mean_degree_nonoverlap")[0]
            rho = group_mean(sub, "spearman_degree_vs_strict_correct")[0]
            lines.append(f"  {blend:<7}{do:>13.2f}{dn:>16.2f}{rho:>26.3f}")

    lines.append("\n\n### Mechanism 4: edge-only vs. content-only ablation (strict_acc_overlap, lenient_acc_overlap, onmi_standard)")
    lines.append("=" * 100)
    for rate in RATES:
        lines.append(f"\noverlap_rate={rate}")
        for blend in BLENDS:
            if blend == 0.0:
                continue
            lines.append(f"  blend={blend}")
            lines.append(f"    {'condition':<14}{'strict_ov':>11}{'lenient_ov':>12}{'onmi_std':>10}")
            for cond in CONDITIONS_NONZERO:
                sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                       and r["condition"] == cond and not r["error"]]
                if not sub:
                    continue
                so = group_mean(sub, "strict_acc_overlap")[0]
                lo = group_mean(sub, "lenient_acc_overlap")[0]
                onmi = group_mean(sub, "onmi_standard")[0]
                lines.append(f"    {cond:<14}{so:>11.3f}{lo:>12.3f}{onmi:>10.3f}")

    atomic_write_text(SUMMARY_LOG, "\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_LOG}")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        ax = axes[0, 0]
        for rate in RATES:
            xs, y_onmi, y_strict, y_lenient = [], [], [], []
            for blend in BLENDS:
                sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                       and r["condition"] == "combined" and not r["error"]]
                if not sub:
                    continue
                xs.append(blend)
                y_onmi.append(group_mean(sub, "onmi_standard")[0])
                y_strict.append(group_mean(sub, "strict_acc_overlap")[0])
                y_lenient.append(group_mean(sub, "lenient_acc_overlap")[0])
            ax.plot(xs, y_onmi, "-o", label=f"onmi_standard r={rate}")
            ax.plot(xs, y_strict, "--^", label=f"strict_acc_overlap r={rate}")
            ax.plot(xs, y_lenient, ":s", label=f"lenient_acc_overlap r={rate}")
        ax.set_xlabel("blend_strength"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
        ax.set_title("Mechanism 2/3: standard ONMI vs. strict/lenient accuracy\n(overlapping nodes only)")
        ax.legend(fontsize=6)

        ax = axes[0, 1]
        for rate in RATES:
            xs, y_c1, y_c2, y_nei = [], [], [], []
            for blend in BLENDS:
                sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                       and r["condition"] == "combined" and not r["error"]]
                if not sub:
                    continue
                xs.append(blend)
                y_c1.append(group_mean(sub, "frac_hard_c1")[0])
                y_c2.append(group_mean(sub, "frac_hard_c2")[0])
                y_nei.append(group_mean(sub, "frac_hard_neither")[0])
            ax.plot(xs, y_c1, "-o", label=f"-> c1 (primary) r={rate}")
            ax.plot(xs, y_c2, "-s", label=f"-> c2 (secondary) r={rate}")
            ax.plot(xs, y_nei, "-^", label=f"-> neither r={rate}")
        ax.set_xlabel("blend_strength"); ax.set_ylabel("fraction of overlapping nodes"); ax.set_ylim(0, 1)
        ax.set_title("Where overlapping nodes' hard assignment lands")
        ax.legend(fontsize=6)

        ax = axes[1, 0]
        for rate in RATES:
            xs, y_do, y_dn = [], [], []
            for blend in BLENDS:
                sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                       and r["condition"] == "combined" and not r["error"]]
                if not sub:
                    continue
                xs.append(blend)
                y_do.append(group_mean(sub, "mean_degree_overlap")[0])
                y_dn.append(group_mean(sub, "mean_degree_nonoverlap")[0])
            ax.plot(xs, y_do, "-o", label=f"overlap nodes r={rate}")
            ax.plot(xs, y_dn, "--s", label=f"non-overlap nodes r={rate}")
        ax.set_xlabel("blend_strength"); ax.set_ylabel("mean degree")
        ax.set_title("Mechanism 1: degree by node group")
        ax.legend(fontsize=6)

        ax = axes[1, 1]
        for rate in RATES:
            for cond, style in (("combined", "-o"), ("edge_only", "--^"), ("content_only", ":s")):
                xs, ys = [], []
                for blend in BLENDS:
                    if blend == 0.0 and cond != "combined":
                        continue
                    sub = [r for r in rows if _fnum(r["rate"]) == rate and _fnum(r["blend"]) == blend
                           and r["condition"] == cond and not r["error"]]
                    if not sub:
                        continue
                    xs.append(blend)
                    ys.append(group_mean(sub, "onmi_standard")[0])
                if xs:
                    ax.plot(xs, ys, style, label=f"{cond} r={rate}", markersize=5)
        ax.set_xlabel("blend_strength"); ax.set_ylabel("onmi_standard"); ax.set_ylim(0, 1)
        ax.set_title("Mechanism 4: edge-only vs. content-only ablation")
        ax.legend(fontsize=6)

        fig.tight_layout()
        fig.savefig(DIAG_PNG, dpi=130)
        plt.close(fig)
        print(f"wrote {DIAG_PNG}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot failed: {exc}")


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
