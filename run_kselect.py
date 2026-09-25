"""
k-selection sensitivity and unsupervised heuristics for NF-MCD.

README "Next steps" item 5: every prior experiment searched k only in a
narrow window around the KNOWN ground-truth community count. This script
asks (a) how costly it is to get k wrong, and (b) whether any label-free
heuristic reliably recovers a good k.

Usage (from nfmcd_impl/):
    py run_kselect.py heuristics          # FPC / eigengap / silhouette (label-free, cheap, deterministic)
    py run_kselect.py run [--workers N]   # k-sweep fits, Part A + Part C (resumable)
    py run_kselect.py summary             # tables + plots, combines run + heuristics output

Part A: for crisismmd, fakeddit, fakeddit_real_lcc, pheme, dblp, amazon, fit
NF-MCD default AND NFMCD.robust() at every k in range(2, k_true+8) (capped at
20), 3 seeds. structural_dim tracks k (NFMCD's own default), so the structural
embedding dimension varies with k too -- that is what a real user sweeping k
would get, not an artifact of this script.

Part B: four label-free k-selection heuristics, evaluated by looking up what
Part A actually achieved at each heuristic's chosen k, vs at k_true:
  1. select_k_by_fpc (nf_mcd.community_detection, already existed, unused
     until now) on a FIXED feature matrix (16-dim structural embedding,
     16-dim raw-PCA content if the dataset has content, concatenated,
     unweighted -- no alpha, since alpha itself depends on a fitted model).
  2. Eigengap of the normalized graph Laplacian (structure only).
  3. Modularity scan over the SAME full k-range as Part A (widened from the
     {k-1,k,k+1} window used elsewhere in this project), separately for
     default and robust (reuses Part A fits, no extra cost).
  4. Silhouette score of a k-means hard partition on the same fixed feature
     matrix as (1).

Part C: does k-selection get harder as structure gets noisier? Modularity-
scan and eigengap only (cheaper), on the crisismmd homophily-sweep graphs
(p_out/p_in in {0.02, 0.25, 1.0}, reusing run_realgraph's SBM regenerator),
NF-MCD default only.

Crash-safe: each finished fit is appended to experiments/kselect_results.csv
(flush + fsync); reruns skip finished (part, dataset, variant, k, seed)
combinations. Other files are written temp+rename.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "kselect_results.csv")
SUMMARY_LOG = os.path.join(EXP, "kselect_summary.log")
PROGRESS_MD = os.path.join(EXP, "kselect_progress.md")
HEURISTICS_JSON = os.path.join(EXP, "kselect_heuristics.json")
CURVES_PNG = os.path.join(EXP, "kselect_curves.png")
HEUR_PNG = os.path.join(EXP, "kselect_heuristics.png")

SEEDS = (0, 1, 2)
DATASETS_A = ["crisismmd", "fakeddit", "fakeddit_real_lcc", "pheme", "dblp", "amazon"]
SWEEP_RATIOS = (0.02, 0.25, 1.0)
VARIANTS = ("default", "robust")
FIELDS = ["part", "dataset", "variant", "k", "seed", "k_true", "onmi", "modularity", "f1", "error", "secs"]

FIXED_DIM = 16  # feature dimensionality for FPC / silhouette (independent of any candidate k)


# ---------------------------------------------------------------------------
# io helpers (same pattern as run_realgraph.py / run_clusterers.py)
# ---------------------------------------------------------------------------

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# k-selection experiment progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished work is skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_kselect.py heuristics\n"
        "py run_kselect.py run --workers 4\n"
        "py run_kselect.py summary\n"
        "```\n"
    ))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "part":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("k", "seed", "k_true"):
                    row[k] = int(row[k])
                for k in ("onmi", "modularity", "f1"):
                    row[k] = float(row[k])
            except ValueError:
                continue
            res[(row["part"], row["dataset"], row["variant"], row["k"], row["seed"])] = row
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


# ---------------------------------------------------------------------------
# k ranges
# ---------------------------------------------------------------------------

def base_k_true(base_key):
    from run_realgraph import get_dataset
    return int(get_dataset(base_key, 0)["n_communities"])


def k_values_for(k_true, cap=20):
    k_max = min(k_true + 7, cap)  # range(2, k_true+8), capped
    return list(range(2, k_max + 1))


# ---------------------------------------------------------------------------
# Part A + Part C: k-sweep fits
# ---------------------------------------------------------------------------

def all_jobs():
    jobs = []
    for ds in DATASETS_A:
        k_true = base_k_true(ds)
        for k in k_values_for(k_true):
            for variant in VARIANTS:
                for seed in SEEDS:
                    jobs.append(("A", ds, variant, k, seed, k_true))
    crisis_k_true = base_k_true("crisismmd")
    for r in SWEEP_RATIOS:
        ds = f"crisismmd@r{r}"
        for k in k_values_for(crisis_k_true):
            for seed in SEEDS:
                jobs.append(("C", ds, "default", k, seed, crisis_k_true))
    return jobs


def run_one(job):
    part, ds, variant, k, seed, k_true = job
    t0 = time.monotonic()
    row = dict(part=part, dataset=ds, variant=variant, k=k, seed=seed, k_true=k_true,
               onmi=float("nan"), modularity=float("nan"), f1=float("nan"), error="")
    try:
        from run_realgraph import get_dataset
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD
        d = get_dataset(ds, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if variant == "default":
                model = NFMCD(n_communities=k, seed=seed)
            else:
                model = NFMCD.robust(n_communities=k, seed=seed)
            model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
            res = b._nfmcd_result(model)
            s = b.score(d, res)
        row.update(onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs done; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress("run", n_done, len(jobs), "running")
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
                    if n_err <= 8:
                        print(f"  FAILED {row['part']}/{row['dataset']}/{row['variant']}/k={row['k']}/"
                              f"seed{row['seed']}: {row['error']}", flush=True)
                if n_done % 25 == 0:
                    write_progress("run", n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress("run", n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# Part B: label-free heuristics (FPC, eigengap, silhouette; modularity-scan
# is derived from the Part A results in do_summary, not computed here)
# ---------------------------------------------------------------------------

def fixed_features(d, dim=FIXED_DIM, seed=0):
    from nf_mcd import topology as topo
    Zs = topo.compute_structural_embedding(d["G"], dim=dim, seed=seed)
    has_content = any(v is not None for v in d["e_t"]) or any(v is not None for v in d["e_v"])
    if has_content:
        Zc = topo.raw_pca_content(d["e_t"], d["e_v"], dim=dim, seed=seed)
        return np.hstack([Zc, Zs])
    return Zs


def eigengap_k(G, k_max, seed=0):
    import networkx as nx
    import scipy.sparse.linalg as sla

    nodes = list(G.nodes())
    n = len(nodes)
    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(float)
    kk = min(k_max + 2, n - 1)
    try:
        v0 = np.random.default_rng(seed).normal(size=n)
        vals = sla.eigsh(L, k=kk, which="SM", v0=v0, return_eigenvectors=False)
        vals = np.sort(vals)
    except Exception:
        vals = np.sort(np.linalg.eigvalsh(L.toarray()))[:kk]
    gaps = np.diff(vals)
    lo, hi = 1, min(len(gaps), k_max)
    if hi <= lo:
        return 2, vals.tolist(), gaps.tolist()
    i_star = lo + int(np.argmax(gaps[lo:hi]))
    k_hat = max(2, min(i_star + 1, k_max))
    return k_hat, vals.tolist(), gaps.tolist()


def silhouette_k(X, k_values, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores = {}
    n = X.shape[0]
    for k in k_values:
        if k >= n:
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X, labels))
    if not scores:
        return k_values[0], scores
    k_hat = max(scores, key=scores.get)
    return k_hat, scores


def do_heuristics():
    from run_realgraph import get_dataset
    from nf_mcd import community_detection as cd

    out = {}
    for ds in DATASETS_A:
        d = get_dataset(ds, 0)
        k_true = int(d["n_communities"])
        k_values = k_values_for(k_true)
        k_max = k_values[-1]
        X = fixed_features(d)

        k_fpc = cd.select_k_by_fpc(X, k_range=range(k_values[0], k_max + 1), m=2.0, seed=0)
        k_eig, eigvals, eiggaps = eigengap_k(d["G"], k_max, seed=0)
        k_sil, sil_scores = silhouette_k(X, k_values, seed=0)

        out[ds] = dict(
            k_true=k_true, k_values=k_values, fixed_dim=FIXED_DIM,
            fpc_k=k_fpc, eigengap_k=k_eig, silhouette_k=k_sil,
            silhouette_scores={str(k): v for k, v in sil_scores.items()},
            eigenvalues=eigvals, eigengaps=eiggaps,
        )
        print(f"{ds}: k_true={k_true}  fpc->{k_fpc}  eigengap->{k_eig}  silhouette->{k_sil}", flush=True)

    # Part C: eigengap only, across structure noise (modularity-scan comes from Part A/C fits).
    crisis_k_true = base_k_true("crisismmd")
    k_values = k_values_for(crisis_k_true)
    out["_sweep_eigengap"] = {}
    for r in SWEEP_RATIOS:
        ds = f"crisismmd@r{r}"
        d = get_dataset(ds, 0)
        k_eig, eigvals, eiggaps = eigengap_k(d["G"], k_values[-1], seed=0)
        out["_sweep_eigengap"][str(r)] = dict(eigengap_k=k_eig, k_true=crisis_k_true)
        print(f"{ds}: eigengap->{k_eig} (k_true={crisis_k_true})", flush=True)

    atomic_write_text(HEURISTICS_JSON, json.dumps(out, indent=2))
    print("wrote", HEURISTICS_JSON)


# ---------------------------------------------------------------------------
# Summary: k-sensitivity curves, heuristic-vs-achieved cost tables, plots
# ---------------------------------------------------------------------------

def _agg(res):
    cell = defaultdict(list)  # (part, dataset, variant, k) -> rows
    for (part, ds, variant, k, seed), r in res.items():
        cell[(part, ds, variant, k)].append(r)
    out = {}
    for key, rows in cell.items():
        out[key] = dict(
            onmi=(float(np.mean([r["onmi"] for r in rows])), float(np.std([r["onmi"] for r in rows]))),
            modularity=(float(np.mean([r["modularity"] for r in rows])), float(np.std([r["modularity"] for r in rows]))),
            f1=(float(np.mean([r["f1"] for r in rows])), float(np.std([r["f1"] for r in rows]))),
            n=len(rows),
        )
    return out


def _best_k_by_modularity(agg, part, ds, variant, k_values):
    best_k, best_mod = None, -np.inf
    for k in k_values:
        rec = agg.get((part, ds, variant, k))
        if rec is None:
            continue
        mu = rec["modularity"][0]
        if mu > best_mod:
            best_mod, best_k = mu, k
    return best_k


def do_summary():
    res = load_results()
    if not res:
        print("no results yet; run `py run_kselect.py run` first")
        return
    agg = _agg(res)
    heur = json.load(open(HEURISTICS_JSON)) if os.path.exists(HEURISTICS_JSON) else {}

    L = ["k-selection sensitivity and unsupervised heuristics for NF-MCD", ""]
    L.append("Part A: ONMI/modularity/F1 vs k (mean +- sd over seeds 0,1,2), k in range(2, k_true+8) capped at 20.")
    L.append("structural_dim tracks k (NFMCD's own default), so a real user sweeping k gets this too.")
    L.append("")

    # --- Part A: k-sensitivity curves + where each metric peaks -------------
    for ds in DATASETS_A:
        k_true = base_k_true(ds)
        k_values = k_values_for(k_true)
        L.append(f"=== {ds}  (k_true = {k_true}) ===")
        for variant in VARIANTS:
            header = f"{'k':>4}" + "".join(f"{'ONMI':>12}{'Mod':>10}{'F1':>10}" for _ in [0])
            L.append(f"  [{variant}]  " + header)
            peak_onmi, peak_onmi_k = -np.inf, None
            peak_mod, peak_mod_k = -np.inf, None
            for k in k_values:
                rec = agg.get(("A", ds, variant, k))
                if rec is None:
                    continue
                o, osd = rec["onmi"]
                m, msd = rec["modularity"]
                f1, f1sd = rec["f1"]
                mark = " <-k_true" if k == k_true else ""
                L.append(f"    {k:>2}  onmi={o:.3f}±{osd:.3f}  mod={m:.3f}±{msd:.3f}  f1={f1:.3f}±{f1sd:.3f}{mark}")
                if o > peak_onmi:
                    peak_onmi, peak_onmi_k = o, k
                if m > peak_mod:
                    peak_mod, peak_mod_k = m, k
            L.append(f"    peak ONMI at k={peak_onmi_k} ({peak_onmi:.3f}); peak modularity at k={peak_mod_k} ({peak_mod:.3f})")
        L.append("")

    # --- Part B: heuristics vs k_true, and cost of misspecification --------
    L.append("Part B: label-free heuristic k, vs k_true, vs the cost (ONMI at k_hat vs ONMI at k_true)")
    L.append("-" * 100)
    for ds in DATASETS_A:
        if ds not in heur:
            continue
        h = heur[ds]
        k_true = h["k_true"]
        k_values = h["k_values"]
        L.append(f"{ds}  (k_true={k_true})")
        heuristics = {
            "fpc": h["fpc_k"],
            "eigengap": h["eigengap_k"],
            "silhouette": h["silhouette_k"],
        }
        for variant in VARIANTS:
            heuristics[f"modularity_scan[{variant}]"] = _best_k_by_modularity(agg, "A", ds, variant, k_values)
        for name, k_hat in heuristics.items():
            if k_hat is None:
                L.append(f"    {name:<24} k_hat=n/a")
                continue
            variant = name.split("[")[1][:-1] if "[" in name else "default"
            rec_hat = agg.get(("A", ds, variant, k_hat))
            rec_true = agg.get(("A", ds, variant, k_true))
            if rec_hat is None or rec_true is None:
                L.append(f"    {name:<24} k_hat={k_hat} (missing fit)")
                continue
            o_hat, o_true = rec_hat["onmi"][0], rec_true["onmi"][0]
            L.append(f"    {name:<24} k_hat={k_hat:>3}  off_by={k_hat - k_true:+d}  "
                      f"onmi@k_hat={o_hat:.3f}  onmi@k_true={o_true:.3f}  cost={o_true - o_hat:+.3f}")
        L.append("")

    # --- Part C: structure noise interaction --------------------------------
    L.append("Part C: does k-selection get harder as structure gets noisier? (crisismmd, NF-MCD default only)")
    L.append("-" * 100)
    crisis_k_true = base_k_true("crisismmd")
    k_values = k_values_for(crisis_k_true)
    sweep_eig = heur.get("_sweep_eigengap", {})
    for r in SWEEP_RATIOS:
        ds = f"crisismmd@r{r}"
        best_mod_k = _best_k_by_modularity(agg, "C", ds, "default", k_values)
        eig = sweep_eig.get(str(r), {})
        k_eig = eig.get("eigengap_k")
        L.append(f"  p_out/p_in={r}:")
        for name, k_hat in (("modularity_scan", best_mod_k), ("eigengap", k_eig)):
            if k_hat is None:
                L.append(f"    {name:<18} k_hat=n/a")
                continue
            rec_hat = agg.get(("C", ds, "default", k_hat))
            rec_true = agg.get(("C", ds, "default", crisis_k_true))
            if rec_hat is None or rec_true is None:
                L.append(f"    {name:<18} k_hat={k_hat} (missing fit)")
                continue
            o_hat, o_true = rec_hat["onmi"][0], rec_true["onmi"][0]
            L.append(f"    {name:<18} k_hat={k_hat:>3}  off_by={k_hat - crisis_k_true:+d}  "
                      f"onmi@k_hat={o_hat:.3f}  onmi@k_true={o_true:.3f}  cost={o_true - o_hat:+.3f}")
    L.append("")

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    do_plots(agg, heur)


def do_plots(agg, heur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, ds in zip(axes.flat, DATASETS_A):
        k_true = base_k_true(ds)
        k_values = k_values_for(k_true)
        for variant, color in (("default", "#888888"), ("robust", "#ff7f00")):
            ys = [agg[("A", ds, variant, k)]["onmi"][0] if ("A", ds, variant, k) in agg else np.nan for k in k_values]
            ax.plot(k_values, ys, marker="o", ms=3, color=color, label=variant)
        ax.axvline(k_true, color="black", ls="--", lw=1, alpha=0.6, label="k_true" if ds == DATASETS_A[0] else None)
        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("LFK ONMI")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CURVES_PNG + ".tmp.png", dpi=130)
    plt.close(fig)
    os.replace(CURVES_PNG + ".tmp.png", CURVES_PNG)

    # heuristic chosen-k vs k_true, bar per dataset
    fig, ax = plt.subplots(figsize=(11, 5))
    names = ["fpc", "eigengap", "silhouette", "modularity_scan[default]", "modularity_scan[robust]"]
    width = 0.15
    xs = np.arange(len(DATASETS_A))
    for i, name in enumerate(names):
        vals = []
        for ds in DATASETS_A:
            h = heur.get(ds, {})
            if name == "fpc":
                v = h.get("fpc_k")
            elif name == "eigengap":
                v = h.get("eigengap_k")
            elif name == "silhouette":
                v = h.get("silhouette_k")
            else:
                variant = name.split("[")[1][:-1]
                v = _best_k_by_modularity(agg, "A", ds, variant, k_values_for(base_k_true(ds)))
            vals.append(v if v is not None else 0)
        ax.bar(xs + i * width, vals, width=width, label=name)
    truevals = [base_k_true(ds) for ds in DATASETS_A]
    ax.scatter(xs + 2 * width, truevals, color="black", marker="*", s=120, zorder=5, label="k_true")
    ax.set_xticks(xs + 2 * width)
    ax.set_xticklabels(DATASETS_A, rotation=20)
    ax.set_ylabel("chosen k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(HEUR_PNG + ".tmp.png", dpi=130)
    plt.close(fig)
    os.replace(HEUR_PNG + ".tmp.png", HEUR_PNG)
    print("saved", CURVES_PNG, "and", HEUR_PNG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["heuristics", "run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "heuristics":
        write_progress("heuristics", 0, 1, "computing FPC / eigengap / silhouette")
        do_heuristics()
        write_progress("heuristics", 1, 1, "heuristics complete; next: run")
    elif args.cmd == "run":
        do_run(args.workers)
    else:
        do_summary()


if __name__ == "__main__":
    main()
