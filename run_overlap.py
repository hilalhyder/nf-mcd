"""
Does NF-MCD's soft membership matrix U actually identify which nodes truly
belong to more than one community? Every prior experiment scored NF-MCD via
overlapping_nmi / membership_f1 at a single fixed overlap_threshold=0.2, but
never asked this directly -- the method's central pitch (fuzzy, overlapping
detection vs. hard partitioning) had never been tested on that specific claim.

CORRECTION vs. the original directive: the directive assumed crisismmd and
fakeddit have ~10% true overlap built into their loaders. Verified from
nf_mcd/datasets.py source: load_crisismmd, load_fakeddit and load_pheme all
build `true_communities = [{int(c)} for c in ...]` -- singleton sets, ZERO
true overlap. Only the synthetic generator (`overlap_rate`) and the SNAP
loaders (dblp: 11/1072 nodes overlap, amazon: 68/146) have real overlap in
ground truth. Part C is reframed accordingly (see its section below): instead
of "does content help correctly identify overlapping nodes" (untestable --
there are no true positives on crisismmd/fakeddit), it measures how often
each content condition CLAIMS a node overlaps when ground truth says every
node belongs to exactly one community (a false-positive overlap rate).

Part A: synthetic ground truth (generate_synthetic_multimodal_graph,
overlap_rate swept), full control.
Part B: real overlap ground truth (dblp, amazon; structure-only, no content).
Part C: reframed content ablation on crisismmd/fakeddit (false-positive rate).

Usage (from nfmcd_impl/):
    py run_overlap.py run [--workers N]
    py run_overlap.py summary

Crash-safe: each finished job appended to experiments/overlap_results.csv and
experiments/overlap_thresholds.csv (flush+fsync); reruns skip finished jobs;
other files written temp+rename.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "overlap_results.csv")
CURVE_CSV = os.path.join(EXP, "overlap_thresholds.csv")
SUMMARY_LOG = os.path.join(EXP, "overlap_summary.log")
PROGRESS_MD = os.path.join(EXP, "overlap_progress.md")
CURVE_PNG = os.path.join(EXP, "overlap_threshold_curve.png")
AUROC_PNG = os.path.join(EXP, "overlap_auroc.png")

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
SYNTH_RATES = (0.0, 0.1, 0.2, 0.3, 0.4)
SYNTH_SEEDS = (0, 1, 2, 3, 4)
REAL_SEEDS = (0, 1, 2)

RESULT_FIELDS = [
    "part", "dataset", "method", "seed", "n_nodes", "true_overlap_rate", "n_true_overlap",
    "auroc", "ap", "best_thr", "best_f1", "precision_at_best", "recall_at_best",
    "f1_at_0.2", "precision_at_0.2", "recall_at_0.2",
    "second_comm_acc", "n_eval_second", "note", "error", "secs",
]
CURVE_FIELDS = ["part", "dataset", "method", "seed", "threshold", "precision", "recall", "f1"]


# ---------------------------------------------------------------------------
# io helpers (same pattern as run_clusterers.py / run_kselect.py)
# ---------------------------------------------------------------------------

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_row(path, fields, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_done(path, key_fields):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(tuple(row[k] for k in key_fields))
    return done


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Overlap-detection experiment progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_overlap.py run --workers 4\n"
        "py run_overlap.py summary\n"
        "```\n"
    ))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def binary_prf(true_bin: np.ndarray, pred_bin: np.ndarray):
    tp = int(np.sum(true_bin & pred_bin))
    fp = int(np.sum(~true_bin & pred_bin))
    fn = int(np.sum(true_bin & ~pred_bin))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def hungarian_align(hard_pred: np.ndarray, true_primary: np.ndarray, k: int):
    from scipy.optimize import linear_sum_assignment
    conf = np.zeros((k, k))
    for p, t in zip(hard_pred, true_primary):
        conf[int(p), int(t)] += 1
    row, col = linear_sum_assignment(-conf)
    return {int(r): int(c) for r, c in zip(row, col)}


def overlap_curve_and_summary(U: np.ndarray, true_bin: np.ndarray):
    """Threshold sweep (precision/recall/F1) + AUROC/AP from the 2nd-highest
    membership as a continuous overlap score. Returns (curve_rows, summary_dict)."""
    from nf_mcd import community_detection as cd
    curve = []
    best = (None, -1.0, 0.0, 0.0)  # thr, f1, precision, recall
    for thr in THRESHOLDS:
        pred_bin = np.array([len(s) > 1 for s in cd.overlapping_communities(U, thr)])
        p, r, f1 = binary_prf(true_bin, pred_bin)
        curve.append((thr, p, r, f1))
        if f1 > best[1]:
            best = (thr, f1, p, r)

    second_highest = np.sort(U, axis=1)[:, -2]
    auroc = ap = float("nan")
    if 0 < true_bin.sum() < len(true_bin):
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(true_bin, second_highest))
        ap = float(average_precision_score(true_bin, second_highest))

    at02 = next(c for c in curve if c[0] == 0.20)
    return curve, dict(
        auroc=auroc, ap=ap, best_thr=best[0], best_f1=best[1],
        precision_at_best=best[2], recall_at_best=best[3],
        **{"f1_at_0.2": at02[3], "precision_at_0.2": at02[1], "recall_at_0.2": at02[2]},
    )


def second_community_accuracy(U: np.ndarray, hard: np.ndarray, true_primary: np.ndarray,
                               true_communities, k: int):
    """Among truly-overlapping nodes: does the model's 2nd-highest-membership
    community, mapped onto true community ids via Hungarian alignment of the
    hard partition to true_primary, match the node's actual second true
    community? Returns (accuracy, n_evaluated) or (nan, 0) if none overlap."""
    mapping = hungarian_align(hard, true_primary, k)
    second_idx = np.argsort(-U, axis=1)[:, 1]
    hits, n = 0, 0
    for i, comms in enumerate(true_communities):
        if len(comms) <= 1:
            continue
        true_second = next((c for c in comms if c != true_primary[i]), None)
        if true_second is None:
            continue
        n += 1
        pred_second_true_id = mapping.get(int(second_idx[i]), -1)
        if pred_second_true_id == true_second:
            hits += 1
    return (hits / n if n else float("nan")), n


def synthetic_primary(n_nodes: int, n_communities: int) -> np.ndarray:
    """Recomputes nf_mcd.datasets.generate_synthetic_multimodal_graph's internal
    `primary` block assignment (deterministic block sizing, not returned by the
    dataclass) purely from n_nodes/n_communities -- no randomness, matches the
    source exactly. Needed for Hungarian alignment / second-community identity."""
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes - sum(sizes)
    primary = []
    for c, size in enumerate(sizes):
        primary.extend([c] * size)
    return np.array(primary[:n_nodes])


# ---------------------------------------------------------------------------
# Part A: synthetic
# ---------------------------------------------------------------------------

def _load_pkl(name):
    with open(os.path.join(CACHE, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def job_partA(rate: float, seed: int):
    """One synthetic dataset (rate, seed): default + robust NF-MCD, DEMON, SLPA,
    two naive calibration baselines. Returns list of (result_row, curve_rows)."""
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    from nf_mcd.pipeline import NFMCD
    from nf_mcd import baselines as blib

    out = []
    key = f"r{rate}"
    data = generate_synthetic_multimodal_graph(
        n_nodes=300, n_communities=5, overlap_rate=rate, seed=1000 + seed,
    )
    n = data.G.number_of_nodes()
    true_bin = np.array([len(t) > 1 for t in data.true_communities])
    true_primary = synthetic_primary(n, data.n_communities)
    base = dict(part="A", dataset=key, seed=seed, n_nodes=n,
                true_overlap_rate=round(float(true_bin.mean()), 4), n_true_overlap=int(true_bin.sum()))

    for method, ctor in (
        ("nfmcd_default", lambda: NFMCD(n_communities=data.n_communities, seed=seed)),
        ("nfmcd_robust", lambda: NFMCD.robust(n_communities=data.n_communities, seed=seed)),
    ):
        t0 = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ctor().fit(data.G, text_embeddings=data.text_embeddings, image_embeddings=data.image_embeddings)
        U, hard = model.U_, model.predict_hard()
        curve, summ = overlap_curve_and_summary(U, true_bin)
        acc, n_eval = second_community_accuracy(U, hard, true_primary, data.true_communities, data.n_communities)
        row = dict(base, method=method, second_comm_acc=round(acc, 4) if n_eval else "",
                   n_eval_second=n_eval, note="", error="", secs=round(time.monotonic() - t0, 2), **summ)
        out.append((row, [dict(part="A", dataset=key, seed=seed, method=method, threshold=t, precision=p, recall=r, f1=f)
                           for t, p, r, f in curve]))

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
                       note="single fixed operating point, not threshold-based", error="",
                       secs=round(time.monotonic() - t0, 2))
        except Exception as exc:  # noqa: BLE001
            row = dict(base, method=method, auroc="", ap="", best_thr="", best_f1="",
                       precision_at_best="", recall_at_best="",
                       **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                       second_comm_acc="", n_eval_second="", note="",
                       error=f"{type(exc).__name__}: {exc}".replace("\n", " "),
                       secs=round(time.monotonic() - t0, 2))
        out.append((row, []))

    p_always, r_always, f1_always = binary_prf(true_bin, np.ones(n, dtype=bool))
    p_never, r_never, f1_never = binary_prf(true_bin, np.zeros(n, dtype=bool))
    for method, (p, r, f1), note in (
        ("always_overlap (= naive top-2-FCM-center: every node forced to a 2nd community)",
         (p_always, r_always, f1_always), "single fixed operating point"),
        ("always_no_overlap", (p_never, r_never, f1_never), "single fixed operating point"),
    ):
        out.append((dict(base, method=method, auroc="", ap="", best_thr="", best_f1=round(f1, 4),
                          precision_at_best=round(p, 4), recall_at_best=round(r, 4),
                          **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                          second_comm_acc="", n_eval_second="", note=note, error="", secs=0.0), []))
    return out


# ---------------------------------------------------------------------------
# Part B: real overlap ground truth (dblp, amazon; structure-only)
# ---------------------------------------------------------------------------

def job_partB(dataset: str, seed: int):
    from nf_mcd.pipeline import NFMCD
    from nf_mcd import baselines as blib

    d = _load_pkl(dataset)
    n = d["G"].number_of_nodes()
    true_bin = np.array([len(t) > 1 for t in d["true"]])
    base = dict(part="B", dataset=dataset, seed=seed, n_nodes=n,
                true_overlap_rate=round(float(true_bin.mean()), 4), n_true_overlap=int(true_bin.sum()))
    out = []

    for method, ctor in (
        ("nfmcd_default", lambda: NFMCD(n_communities=d["n_communities"], seed=seed)),
        ("nfmcd_robust", lambda: NFMCD.robust(n_communities=d["n_communities"], seed=seed)),
    ):
        t0 = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ctor().fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
        curve, summ = overlap_curve_and_summary(model.U_, true_bin)
        row = dict(base, method=method, second_comm_acc="", n_eval_second="",
                   note="structure-only dataset, no content", error="",
                   secs=round(time.monotonic() - t0, 2), **summ)
        out.append((row, [dict(part="B", dataset=dataset, seed=seed, method=method,
                                threshold=t, precision=p, recall=r, f1=f) for t, p, r, f in curve]))

    for method, fn in (("demon", blib.demon), ("slpa", blib.slpa)):
        t0 = time.monotonic()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fn(d, seed)
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
                       note="single fixed operating point", error="", secs=round(time.monotonic() - t0, 2))
        except Exception as exc:  # noqa: BLE001
            row = dict(base, method=method, auroc="", ap="", best_thr="", best_f1="",
                       precision_at_best="", recall_at_best="",
                       **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                       second_comm_acc="", n_eval_second="", note="",
                       error=f"{type(exc).__name__}: {exc}".replace("\n", " "),
                       secs=round(time.monotonic() - t0, 2))
        out.append((row, []))

    p_always, r_always, f1_always = binary_prf(true_bin, np.ones(n, dtype=bool))
    p_never, r_never, f1_never = binary_prf(true_bin, np.zeros(n, dtype=bool))
    for method, (p, r, f1) in (("always_overlap", (p_always, r_always, f1_always)),
                                ("always_no_overlap", (p_never, r_never, f1_never))):
        out.append((dict(base, method=method, auroc="", ap="", best_thr="", best_f1=round(f1, 4),
                          precision_at_best=round(p, 4), recall_at_best=round(r, 4),
                          **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                          second_comm_acc="", n_eval_second="", note="single fixed operating point",
                          error="", secs=0.0), []))
    return out


# ---------------------------------------------------------------------------
# Part C: reframed -- false-positive overlap rate under content ablation
# (crisismmd, fakeddit have ZERO true overlap; see module docstring)
# ---------------------------------------------------------------------------

def job_partC(dataset: str, seed: int):
    from nf_mcd import baselines as blib

    d = _load_pkl(dataset)
    n = d["G"].number_of_nodes()
    true_bin = np.zeros(n, dtype=bool)  # every node has exactly one true community here
    base = dict(part="C", dataset=dataset, seed=seed, n_nodes=n, true_overlap_rate=0.0, n_true_overlap=0)
    out = []
    for method, text, image in (
        ("nfmcd_default", True, True),
        ("nfmcd_structure_only", False, False),
        ("nfmcd_text_only", True, False),
        ("nfmcd_image_only", False, True),
    ):
        t0 = time.monotonic()
        try:
            model = blib._fit_nfmcd(d, d["n_communities"], seed, text=text, image=image)
        except Exception as exc:  # noqa: BLE001
            out.append((dict(base, method=method, auroc="", ap="", best_thr="", best_f1="",
                              precision_at_best="", recall_at_best="",
                              **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                              second_comm_acc="", n_eval_second="", note="content not available for this method/dataset",
                              error=f"{type(exc).__name__}: {exc}".replace("\n", " "),
                              secs=round(time.monotonic() - t0, 2)), []))
            continue
        # true_bin is all-False here: precision/recall/F1 against "true overlap" are
        # degenerate (no positives to find, AUROC/AP undefined), so what's reported
        # instead is the FALSE-POSITIVE overlap rate = fraction of nodes claimed to
        # overlap when none truly do, at each threshold (lower is better).
        from nf_mcd import community_detection as cd
        fp_by_thr = {th: float(np.mean([len(s) > 1 for s in cd.overlapping_communities(model.U_, th)]))
                     for th in THRESHOLDS}
        row = dict(base, method=method, auroc="", ap="", best_thr="", best_f1="",
                   precision_at_best="", recall_at_best="",
                   **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": round(fp_by_thr[0.20], 4)},
                   second_comm_acc="", n_eval_second="",
                   note="reframed: no true overlap here; recall_at_0.2 column repurposed as "
                        "false-positive overlap rate at threshold 0.2 (lower = better)",
                   error="", secs=round(time.monotonic() - t0, 2))
        curve_rows = [dict(part="C", dataset=dataset, seed=seed, method=method, threshold=th,
                            precision="", recall="", f1=fp_by_thr[th]) for th in THRESHOLDS]
        out.append((row, curve_rows))
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def all_jobs():
    jobs = [("A", rate, seed) for rate in SYNTH_RATES for seed in SYNTH_SEEDS]
    jobs += [("B", ds, seed) for ds in ("dblp", "amazon") for seed in REAL_SEEDS]
    jobs += [("C", ds, seed) for ds in ("crisismmd", "fakeddit") for seed in REAL_SEEDS]
    return jobs


def run_job(job):
    part, key, seed = job
    try:
        if part == "A":
            return part, key, seed, job_partA(key, seed), None
        if part == "B":
            return part, key, seed, job_partB(key, seed), None
        return part, key, seed, job_partC(key, seed), None
    except Exception as exc:  # noqa: BLE001
        return part, key, seed, None, f"{type(exc).__name__}: {exc}".replace("\n", " ")


def _job_key(part, key, seed):
    return (part, str(key), str(seed))


def cmd_run(workers):
    jobs = all_jobs()
    done_keys = load_done(RESULTS_CSV, ["part", "dataset", "seed"]) if os.path.exists(RESULTS_CSV) else set()
    # dataset column stores the composite key (e.g. "r0.1" for part A); use (part,dataset,seed) triples
    # already present as *any* method for that (part,key,seed) -> consider the whole job done.
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
            part, key, seed, rows, err = fut.result()
            if err:
                print(f"  FAILED {part} {key} seed={seed}: {err}")
                append_row(RESULTS_CSV, RESULT_FIELDS, dict(
                    part=part, dataset=key, method="ALL", seed=seed, n_nodes="", true_overlap_rate="",
                    n_true_overlap="", auroc="", ap="", best_thr="", best_f1="", precision_at_best="",
                    recall_at_best="", **{"f1_at_0.2": "", "precision_at_0.2": "", "recall_at_0.2": ""},
                    second_comm_acc="", n_eval_second="", note="", error=err, secs=0))
            else:
                for row, curve_rows in rows:
                    append_row(RESULTS_CSV, RESULT_FIELDS, row)
                    for cr in curve_rows:
                        append_row(CURVE_CSV, CURVE_FIELDS, cr)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(jobs):
                write_progress("run", n_done, len(jobs), f"last: {part} {key} seed={seed}")
    write_progress("run", len(jobs), len(jobs), "all done")
    print("done")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

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


def cmd_summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(RESULTS_CSV)
    curve_rows = _read_csv(CURVE_CSV)
    lines = []

    def group_mean(rows_subset, field):
        vals = [_fnum(r[field]) for r in rows_subset if r[field] not in ("", None)]
        vals = [v for v in vals if not np.isnan(v)]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)

    lines.append("Part A: synthetic overlap detection (n_nodes=300, k=5, 5 seeds)")
    lines.append("=" * 78)
    methods_a = ["nfmcd_default", "nfmcd_robust", "demon", "slpa",
                 "always_overlap (= naive top-2-FCM-center: every node forced to a 2nd community)",
                 "always_no_overlap"]
    for rate in SYNTH_RATES:
        ds = f"r{rate}"
        sub = [r for r in rows if r["part"] == "A" and r["dataset"] == ds]
        if not sub:
            continue
        true_rate = group_mean(sub, "true_overlap_rate")[0]
        lines.append(f"\noverlap_rate={rate}  (observed true overlap rate: {true_rate:.3f})")
        lines.append(f"{'method':<55}{'AUROC':>8}{'AP':>8}{'best_thr':>9}{'best_F1':>9}{'F1@0.2':>8}{'2nd-comm acc':>14}")
        for m in methods_a:
            ms = [r for r in sub if r["method"] == m]
            if not ms:
                continue
            auroc = group_mean(ms, "auroc")[0]
            ap = group_mean(ms, "ap")[0]
            bthr = group_mean(ms, "best_thr")[0]
            bf1 = group_mean(ms, "best_f1")[0]
            f102 = group_mean(ms, "f1_at_0.2")[0]
            acc = group_mean(ms, "second_comm_acc")[0]
            def fmt(x):
                return f"{x:.3f}" if not np.isnan(x) else "-"
            lines.append(f"{m[:54]:<55}{fmt(auroc):>8}{fmt(ap):>8}{fmt(bthr):>9}{fmt(bf1):>9}{fmt(f102):>8}{fmt(acc):>14}")

    lines.append("\n\nPart A: threshold sensitivity (is 0.2 a good default?), nfmcd_default only, mean best_thr vs 0.2")
    lines.append("-" * 78)
    for rate in SYNTH_RATES:
        ds = f"r{rate}"
        sub = [r for r in rows if r["part"] == "A" and r["dataset"] == ds and r["method"] == "nfmcd_default"]
        if not sub:
            continue
        bthr_mean, bthr_sd, _ = group_mean(sub, "best_thr")
        f1_02_mean, _, _ = group_mean(sub, "f1_at_0.2")
        best_f1_mean, _, _ = group_mean(sub, "best_f1")
        lines.append(f"  rate={rate}: argmax-F1 threshold = {bthr_mean:.3f} (sd {bthr_sd:.3f}), "
                      f"F1 there = {best_f1_mean:.3f} vs F1 at fixed 0.2 = {f1_02_mean:.3f}")

    lines.append("\n\nPart B: real overlap ground truth (structure-only, no content)")
    lines.append("=" * 78)
    for ds in ("dblp", "amazon"):
        sub = [r for r in rows if r["part"] == "B" and r["dataset"] == ds]
        if not sub:
            continue
        true_rate = group_mean(sub, "true_overlap_rate")[0]
        n_true = sub[0]["n_true_overlap"]
        n_nodes = sub[0]["n_nodes"]
        lines.append(f"\n{ds}: true overlap rate {true_rate:.3f} ({n_true}/{n_nodes} nodes)")
        lines.append(f"{'method':<20}{'AUROC':>8}{'AP':>8}{'best_thr':>9}{'best_F1':>9}{'F1@0.2':>8}")
        for m in ("nfmcd_default", "nfmcd_robust", "demon", "slpa", "always_overlap", "always_no_overlap"):
            ms = [r for r in sub if r["method"] == m]
            if not ms:
                continue
            auroc = group_mean(ms, "auroc")[0]
            ap = group_mean(ms, "ap")[0]
            bthr = group_mean(ms, "best_thr")[0]
            bf1 = group_mean(ms, "best_f1")[0]
            f102 = group_mean(ms, "f1_at_0.2")[0]
            def fmt(x):
                return f"{x:.3f}" if not np.isnan(x) else "-"
            lines.append(f"{m:<20}{fmt(auroc):>8}{fmt(ap):>8}{fmt(bthr):>9}{fmt(bf1):>9}{fmt(f102):>8}")

    lines.append("\n\nPart C (REFRAMED): false-positive overlap rate under content ablation")
    lines.append("crisismmd/fakeddit have ZERO true overlap (verified from source) -- there is nothing to")
    lines.append("correctly detect, so this measures how often each content condition WRONGLY claims a")
    lines.append("node overlaps at the standard threshold 0.2. Lower is better; 0.000 = never wrong.")
    lines.append("=" * 78)
    for ds in ("crisismmd", "fakeddit"):
        sub = [r for r in rows if r["part"] == "C" and r["dataset"] == ds]
        if not sub:
            continue
        lines.append(f"\n{ds}: false-positive overlap rate @ threshold 0.2, mean over 3 seeds")
        for m in ("nfmcd_default", "nfmcd_structure_only", "nfmcd_text_only", "nfmcd_image_only"):
            ms = [r for r in sub if r["method"] == m]
            if not ms:
                continue
            fp_mean, fp_sd, n = group_mean(ms, "recall_at_0.2")  # repurposed column, see docstring
            lines.append(f"  {m:<25} {fp_mean:.4f} +- {fp_sd:.4f}  (n={n})")

    atomic_write_text(SUMMARY_LOG, "\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_LOG}")

    # -- plots ---------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, len(SYNTH_RATES), figsize=(4 * len(SYNTH_RATES), 3.6), sharey=True)
        for ax, rate in zip(axes, SYNTH_RATES):
            ds = f"r{rate}"
            for method, style in (("nfmcd_default", "-o"), ("nfmcd_robust", "-s")):
                cs = [c for c in curve_rows if c["part"] == "A" and c["dataset"] == ds and c["method"] == method]
                by_thr = {}
                for c in cs:
                    by_thr.setdefault(_fnum(c["threshold"]), []).append(_fnum(c["f1"]))
                xs = sorted(by_thr)
                ys = [np.mean(by_thr[x]) for x in xs]
                if xs:
                    ax.plot(xs, ys, style, label=method, markersize=4)
            ax.axvline(0.2, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"overlap_rate={rate}")
            ax.set_xlabel("threshold")
        axes[0].set_ylabel("F1 (predicting true overlap)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Part A: F1 of overlap detection vs. threshold (dotted = default 0.2)")
        fig.tight_layout()
        fig.savefig(CURVE_PNG, dpi=130)
        plt.close(fig)
        print(f"wrote {CURVE_PNG}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot 1 failed: {exc}")

    try:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels, vals = [], []
        for rate in SYNTH_RATES:
            ds = f"r{rate}"
            for method in ("nfmcd_default", "nfmcd_robust"):
                sub = [r for r in rows if r["part"] == "A" and r["dataset"] == ds and r["method"] == method]
                v = group_mean(sub, "auroc")[0]
                labels.append(f"synth r={rate}\n{method.replace('nfmcd_', '')}")
                vals.append(v)
        for ds in ("dblp", "amazon"):
            for method in ("nfmcd_default", "nfmcd_robust"):
                sub = [r for r in rows if r["part"] == "B" and r["dataset"] == ds and r["method"] == method]
                v = group_mean(sub, "auroc")[0]
                labels.append(f"{ds}\n{method.replace('nfmcd_', '')}")
                vals.append(v)
        x = np.arange(len(labels))
        ax.bar(x, vals, color="#4c72b0")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUROC 0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("AUROC (2nd-highest membership predicting true overlap)")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(AUROC_PNG, dpi=130)
        plt.close(fig)
        print(f"wrote {AUROC_PNG}")
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
