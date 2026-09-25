"""
Run NF-MCD ablations and baselines (nf_mcd/baselines.py) on the five cached
datasets and summarise them.

Usage (from nfmcd_impl/):
    py run_baselines.py run [--workers N]   # run missing (method, dataset, seed) jobs
    py run_baselines.py summary             # rebuild experiments/baselines_summary.log from the CSV

Crash-safe and resumable: each finished job is appended to
experiments/baselines_results.csv and fsynced; on restart, finished jobs are
skipped. Files are written temp-then-rename.

Protocol: k = ground-truth community count for methods that take k (Louvain,
DEMON and SLPA choose their own); nfmcd_scan_k picks its own k by modularity.
Seeds 0,1,2. Metrics: modularity of the hard partition, LFK overlapping NMI
(cdlib), best-match membership F1. Overlap threshold 0.2.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import pickle
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "baselines_results.csv")
SUMMARY_LOG = os.path.join(EXP, "baselines_summary.log")
PROGRESS_MD = os.path.join(EXP, "baselines_progress.md")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
SEEDS = (0, 1, 2)
FIELDS = ["method", "dataset", "seed", "k_used", "n_pred", "modularity", "onmi", "f1", "note", "error", "secs"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Baselines progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_baselines.py run --workers 4\n"
        "py run_baselines.py summary\n"
        "```\n\n"
        "Results: experiments/baselines_results.csv (append-only, fsynced per job). "
        "Summary: experiments/baselines_summary.log.\n"
    ))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "method":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("modularity", "onmi", "f1"):
                    row[k] = float(row[k])
                row["seed"] = int(row["seed"])
            except ValueError:
                continue
            res[(row["method"], row["dataset"], row["seed"])] = row
    return res


def open_for_append():
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new:
        with open(RESULTS_CSV, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_nl = f.read(1) == b"\n"
        if not ends_nl:
            with open(RESULTS_CSV, "ab") as f:
                f.write(b"\n")
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(FIELDS)
        f.flush()
        os.fsync(f.fileno())
    return f, w


_CACHE = {}


def run_one(job):
    method, ds, seed = job
    t0 = time.monotonic()
    row = dict(method=method, dataset=ds, seed=seed, k_used="", n_pred="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"), note="", error="")
    try:
        from nf_mcd import baselines as b
        if ds not in _CACHE:
            with open(os.path.join(CACHE, f"{ds}.pkl"), "rb") as f:
                _CACHE[ds] = pickle.load(f)
        d = _CACHE[ds]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = b.METHODS[method][0](d, seed)
            s = b.score(d, res)
        row.update(k_used=res.k_used, n_pred=s["n_pred"], modularity=s["modularity"],
                   onmi=s["onmi"], f1=s["f1"], note=res.note)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    from nf_mcd import baselines as b
    return [(m, ds, s) for m, (_, app) in b.METHODS.items() for ds in DATASETS if ds in app for s in SEEDS]


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if j not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress(n_done, len(jobs), "running")
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(run_one, j) for j in pending]):
                row = fut.result()
                w.writerow([row[k] for k in FIELDS])
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  FAILED {row['method']}/{row['dataset']}/seed{row['seed']}: {row['error']}", flush=True)
                if n_done % 10 == 0:
                    write_progress(n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


def _fmt(vals):
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f}±{a.std(ddof=0):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


def do_summary():
    from nf_mcd import baselines as b
    res = load_results()
    agg = defaultdict(lambda: defaultdict(list))
    for (m, ds, s), r in res.items():
        for key in ("modularity", "onmi", "f1"):
            agg[(m, ds)][key].append(r[key])
        agg[(m, ds)]["k"].append(r["k_used"])
        agg[(m, ds)]["npred"].append(r["n_pred"])

    methods = list(b.METHODS)
    lines = []

    def table(metric, title):
        lines.append(f"\n{title}")
        lines.append("-" * len(title))
        head = f"{'method':<24}" + "".join(f"{d:>16}" for d in DATASETS)
        lines.append(head)
        for m in methods:
            row = f"{m:<24}"
            for ds in DATASETS:
                if ds not in b.METHODS[m][1]:
                    row += f"{'n/a':>16}"
                elif (m, ds) in agg:
                    row += f"{_fmt(agg[(m, ds)][metric]):>16}"
                else:
                    row += f"{'missing':>16}"
            lines.append(row)

    lines.append("Baselines and ablations (mean±sd over seeds 0,1,2)")
    lines.append("k = ground-truth community count for methods that take k; Louvain/DEMON/SLPA pick their own;")
    lines.append("nfmcd_scan_k picks k by modularity over {k-1,k,k+1}. Overlap threshold 0.2.")
    table("onmi", "Table 1: LFK overlapping NMI")
    table("modularity", "Table 2a: modularity of the hard partition")
    table("f1", "Table 2b: membership F1")

    lines.append("\nCommunities found (mean count) by methods that choose their own k")
    for m in ("louvain", "demon", "slpa", "nfmcd_scan_k"):
        vals = "  ".join(f"{ds}={np.mean([float(x) for x in agg[(m, ds)]['npred']]):.1f}"
                         for ds in DATASETS if (m, ds) in agg)
        lines.append(f"  {m:<14}{vals}")

    # Wins / ties / losses of full NF-MCD (k = truth) vs every other method, on ONMI.
    lines.append("\nFull NF-MCD (nfmcd_full, defaults, k=truth) vs. every other method, by LFK ONMI")
    lines.append("tie if |diff| < max(0.02, sd_a + sd_b); W/T/L = NF-MCD wins/ties/loses")
    tot = dict(W=0, T=0, L=0)
    for ds in DATASETS:
        if ("nfmcd_full", ds) not in agg:
            continue
        base = np.array(agg[("nfmcd_full", ds)]["onmi"])
        wins, ties, losses = [], [], []
        for m in methods:
            if m == "nfmcd_full" or ds not in b.METHODS[m][1] or (m, ds) not in agg:
                continue
            o = np.array(agg[(m, ds)]["onmi"])
            diff = base.mean() - o.mean()
            if abs(diff) < max(0.02, base.std() + o.std()):
                ties.append(m)
            elif diff > 0:
                wins.append(m)
            else:
                losses.append(m)
        tot["W"] += len(wins); tot["T"] += len(ties); tot["L"] += len(losses)
        lines.append(f"  {ds:<10} NF-MCD {base.mean():.3f}:  W={len(wins)} T={len(ties)} L={len(losses)}")
        lines.append(f"      loses to: {', '.join(losses) or '-'}")
        lines.append(f"      ties:     {', '.join(ties) or '-'}")
        lines.append(f"      beats:    {', '.join(wins) or '-'}")
    lines.append(f"  TOTAL W={tot['W']} T={tot['T']} L={tot['L']}")

    lines.append("\nBest method per dataset")
    for metric, label in (("onmi", "ONMI"), ("modularity", "modularity"), ("f1", "F1")):
        parts = []
        for ds in DATASETS:
            cands = [(np.mean(agg[(m, ds)][metric]), m) for m in methods if (m, ds) in agg]
            if cands:
                v, m = max(cands)
                parts.append(f"{ds}: {m} ({v:.3f})")
        lines.append(f"  {label:<11}" + "; ".join(parts))

    text = "\n".join(lines) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        do_run(args.workers)
        do_summary()
    else:
        do_summary()
