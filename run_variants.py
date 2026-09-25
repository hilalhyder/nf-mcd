"""
Optional fusion-variant experiment for NF-MCD (defaults unchanged).

Usage (from nfmcd_impl/):
    py run_variants.py run [--workers N]   # run missing (variant, dataset, graph, seed) jobs
    py run_variants.py summary             # rebuild experiments/variant_summary.log from the CSV
    py run_variants.py verify              # check default variant == baselines nfmcd_full reference

Crash-safe and resumable: each finished job is appended to
experiments/variant_results.csv (flush + fsync); finished jobs are skipped on
restart. Other files are written temp-then-rename.

Protocol: k = ground-truth community count, seeds 0,1,2, overlap threshold 0.2,
metrics = modularity, LFK overlapping NMI (cdlib), membership F1. Fixed variant
list, no tuning. "rewired50" = the Fakeddit/CrisisMMD graph with ~50% of edges
degree-preservingly rewired (networkx double_edge_swap): a robustness check on
label-leaking structure, not a headline result.
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
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "variant_results.csv")
SUMMARY_LOG = os.path.join(EXP, "variant_summary.log")
PROGRESS_MD = os.path.join(EXP, "variant_progress.md")
REFERENCE = os.path.join(EXP, "_default_reference.json")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
REWIRE_DATASETS = ["crisismmd", "fakeddit"]
SEEDS = (0, 1, 2)
FIELDS = ["variant", "dataset", "graph", "seed", "k_used", "modularity", "onmi", "f1", "error", "secs"]

VARIANTS = {
    "default": {},
    "cons_a60": dict(confidence_mode="consistency", alpha_max=0.6),
    "cons_a50": dict(confidence_mode="consistency", alpha_max=0.5),
    "blend_a60": dict(confidence_mode="blend", alpha_max=0.6),
    "agree_a50": dict(alpha_max=0.5),
    "default_rank16": dict(pca_rank_div=16),
    "all_changes": dict(confidence_mode="consistency", alpha_max=0.6, pca_rank_div=16,
                        single_modality_fill="zero"),
}


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Variant experiment progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_variants.py run --workers 4\n"
        "py run_variants.py summary\n"
        "py run_variants.py verify\n"
        "```\n\n"
        "Results: experiments/variant_results.csv (append-only, fsynced per job). "
        "Summary: experiments/variant_summary.log.\n"
    ))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "variant":
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
            res[(row["variant"], row["dataset"], row["graph"], row["seed"])] = row
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


def load_dataset(ds):
    if ds not in _CACHE:
        with open(os.path.join(CACHE, f"{ds}.pkl"), "rb") as f:
            _CACHE[ds] = pickle.load(f)
    return _CACHE[ds]


def rewired_graph(G, seed, frac_edges=0.5):
    import networkx as nx
    H = G.copy()
    m = H.number_of_edges()
    nswap = max(1, int(frac_edges * m / 2))  # each swap rewires 2 edges
    nx.double_edge_swap(H, nswap=nswap, max_tries=nswap * 100, seed=1000 + seed)
    return H


def run_one(job):
    variant, ds, graph, seed = job
    t0 = time.monotonic()
    row = dict(variant=variant, dataset=ds, graph=graph, seed=seed, k_used="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"), error="")
    try:
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD
        d = dict(load_dataset(ds))
        if graph == "rewired50":
            d["G"] = rewired_graph(d["G"], seed)
        k = d["n_communities"]
        n = d["G"].number_of_nodes()
        model = NFMCD(n_communities=k, seed=seed, **VARIANTS[variant])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            res = b._nfmcd_result(model)
            s = b.score(d, res)
        row.update(k_used=res.k_used, modularity=s["modularity"], onmi=s["onmi"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    jobs = [(v, ds, "orig", s) for v in VARIANTS for ds in DATASETS for s in SEEDS]
    jobs += [(v, ds, "rewired50", s) for v in VARIANTS for ds in REWIRE_DATASETS for s in SEEDS]
    return jobs


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if j not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    finished = len(jobs) - len(pending)
    write_progress(finished, len(jobs), "running")

    def record(row):
        nonlocal finished
        w.writerow([row[k] for k in FIELDS])
        f.flush()
        os.fsync(f.fileno())
        if not row["error"]:
            finished += 1
            if finished % 5 == 0:
                write_progress(finished, len(jobs), "running")
        else:
            print("ERROR", row["variant"], row["dataset"], row["graph"], row["seed"], row["error"], flush=True)

    if workers <= 1:
        for j in pending:
            record(run_one(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, j) for j in pending]
            for fu in as_completed(futs):
                record(fu.result())
    f.close()
    write_progress(finished, len(jobs), "run complete" if finished == len(jobs) else "run finished with errors")


def do_verify():
    ref = json.load(open(REFERENCE))
    res = load_results()
    bad, checked = 0, 0
    for key, (onmi, mod, f1) in sorted(ref.items()):
        ds, seed = key.split("|")
        r = res.get(("default", ds, "orig", int(seed)))
        if r is None:
            print("MISSING", key)
            bad += 1
            continue
        checked += 1
        diffs = [abs(r["onmi"] - float(onmi)), abs(r["modularity"] - float(mod)), abs(r["f1"] - float(f1))]
        if max(diffs) > 1e-9:
            bad += 1
            print("MISMATCH", key, "onmi/mod/f1 abs diffs:", diffs)
    print(f"verify: {checked} default rows compared against baselines nfmcd_full, {bad} mismatches/missing")
    return bad


def ms(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def do_summary():
    res = load_results()
    L = []

    def cell(variant, ds, graph, metric="onmi"):
        v = [res[(variant, ds, graph, s)][metric] for s in SEEDS if (variant, ds, graph, s) in res]
        return ms(v) if v else None

    for metric, title in (("onmi", "LFK ONMI"), ("modularity", "Modularity"), ("f1", "Membership F1")):
        L += [f"== {title} (k = ground-truth count, original graph; mean +- sd over seeds 0,1,2) ==",
              f"{'variant':<16}" + "".join(f"{d:>16}" for d in DATASETS)]
        for v in VARIANTS:
            line = f"{v:<16}"
            for ds in DATASETS:
                c = cell(v, ds, "orig", metric)
                line += f"{(f'{c[0]:.3f}+-{c[1]:.3f}' if c else 'n/a'):>16}"
            L.append(line)
        L.append("")

    L += ["== Win / tie / loss vs default (ONMI; tie if |diff| < max(0.02, sd_variant + sd_default)) ==",
          f"{'variant':<16}" + "".join(f"{d:>16}" for d in DATASETS) + f"{'W/T/L':>10}"]
    for v in VARIANTS:
        if v == "default":
            continue
        line, w, t, l = f"{v:<16}", 0, 0, 0
        for ds in DATASETS:
            a, b_ = cell(v, ds, "orig"), cell("default", ds, "orig")
            if not a or not b_:
                line += f"{'n/a':>16}"
                continue
            diff = a[0] - b_[0]
            thr = max(0.02, a[1] + b_[1])
            tag = "WIN" if diff > thr else ("LOSS" if diff < -thr else "tie")
            w += tag == "WIN"; t += tag == "tie"; l += tag == "LOSS"
            line += f"{f'{diff:+.3f} {tag}':>16}"
        L.append(line + f"{f'{w}/{t}/{l}':>10}")
    L.append("")

    L += ["== ROBUSTNESS CHECK (not a headline result): ~50% of edges degree-preservingly rewired ==",
          "ONMI mean+-sd, original graph -> rewired50 graph",
          f"{'variant':<16}" + "".join(f"{d:>28}" for d in REWIRE_DATASETS)]
    for v in VARIANTS:
        line = f"{v:<16}"
        for ds in REWIRE_DATASETS:
            a, b_ = cell(v, ds, "orig"), cell(v, ds, "rewired50")
            s = f"{a[0]:.3f} -> {b_[0]:.3f}+-{b_[1]:.3f}" if a and b_ else "n/a"
            line += f"{s:>28}"
        L.append(line)
    L += ["", "Win/tie/loss vs default on the rewired50 graphs:",
          f"{'variant':<16}" + "".join(f"{d:>20}" for d in REWIRE_DATASETS)]
    for v in VARIANTS:
        if v == "default":
            continue
        line = f"{v:<16}"
        for ds in REWIRE_DATASETS:
            a, b_ = cell(v, ds, "rewired50"), cell("default", ds, "rewired50")
            if not a or not b_:
                line += f"{'n/a':>20}"
                continue
            diff = a[0] - b_[0]
            thr = max(0.02, a[1] + b_[1])
            tag = "WIN" if diff > thr else ("LOSS" if diff < -thr else "tie")
            line += f"{f'{diff:+.3f} {tag}':>20}"
        L.append(line)

    sp = os.path.join(EXP, "variant_spearman.txt")
    if os.path.exists(sp):
        L += ["", "== Consistency confidence vs content-cluster correctness (Spearman) ==", open(sp).read().rstrip()]

    atomic_write_text(SUMMARY_LOG, "\n".join(L) + "\n")
    print("\n".join(L))


def do_spearman():
    """Spearman between each confidence and whether a node's content-only
    k-means cluster (Hungarian-matched to its primary true community) is right."""
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import spearmanr
    from sklearn.cluster import KMeans
    from nf_mcd.pipeline import NFMCD

    lines = []
    for ds in ("crisismmd", "fakeddit", "pheme"):
        d = load_dataset(ds)
        k = d["n_communities"]
        prim = np.array([min(c) if c else -1 for c in d["true"]])
        for mode in ("agreement", "consistency"):
            m = NFMCD(n_communities=k, seed=0, confidence_mode=mode)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            X = m.fused_content_
            valid = (np.linalg.norm(X, axis=1) > 0) & (prim >= 0)
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X[valid])
            cont = np.zeros((k, prim.max() + 1))
            for a, b_ in zip(lab, prim[valid]):
                cont[a, b_] += 1
            r, c = linear_sum_assignment(-cont)
            mp = dict(zip(r, c))
            correct = np.array([mp.get(a, -1) == b_ for a, b_ in zip(lab, prim[valid])], dtype=float)
            conf = m.confidence_[valid]
            rho = spearmanr(conf, correct).correlation if conf.std() > 0 and correct.std() > 0 else float("nan")
            lines.append(f"{ds:<10}{mode:<13} n={int(valid.sum()):>5} mean_conf={conf.mean():.3f} "
                         f"content-cluster accuracy={correct.mean():.3f} spearman(conf, correct)={rho:+.3f}")
    atomic_write_text(os.path.join(EXP, "variant_spearman.txt"), "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "verify", "spearman"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        do_run(a.workers)
    elif a.cmd == "verify":
        sys.exit(1 if do_verify() else 0)
    elif a.cmd == "spearman":
        do_spearman()
    else:
        do_summary()
