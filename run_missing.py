"""
Missing-modality robustness sweep: does NF-MCD degrade more gracefully than
content baselines when text and/or images go missing?

Usage (from nfmcd_impl/):
    py run_missing.py run [--workers N]   # run missing jobs (resumable)
    py run_missing.py summary             # rebuild summary log + plots from the CSV
    py run_missing.py verify              # p=0 rows vs known values

Design
------
* Datasets (cached, no re-encoding): crisismmd, fakeddit, fakeddit_real_lcc
  (real-relations graph, largest component), pheme (text only, so only text
  removal applies).
* MCAR masks with nested draws: per mask seed each node gets two uniforms
  (u_text, u_image); text is removed where u_text < p, image where u_image < p,
  so a node removed at p stays removed at every larger p. Modes: "image"
  (image only), "text" (text only), "both" (each modality independently, so
  P(no content) = p^2). Nodes that already lack a modality in the cache keep
  it missing.
* Methods (identical masks for all): NF-MCD default, NF-MCD all_changes,
  NF-MCD default_rank16 (pca_rank_div=16), spectral graph+content,
  k-means on content; louvain and NF-MCD structure-only never use content, so
  they are run once and replicated across p as constant references.
* Content baselines use whatever modalities remain: L2-normalised text and
  image embeddings concatenated, zero block for a missing modality. K-means
  gets an all-zero content row for nodes with no content. Spectral builds the
  content kNN graph only among nodes that have content (no-content nodes get
  graph edges only).
* k = ground-truth community count. Metrics on ALL nodes: LFK ONMI, hard-
  partition modularity, membership F1.
* 5 mask seeds x 2 method seeds per (dataset, mode, p, method).

Crash-safe: each finished job is appended to experiments/missing_results.csv
and fsynced; restart skips finished jobs. Other files are written temp+rename.
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
RESULTS_CSV = os.path.join(EXP, "missing_results.csv")
SUMMARY_LOG = os.path.join(EXP, "missing_summary.log")
PROGRESS_MD = os.path.join(EXP, "missing_progress.md")

DATASET_MODES = {
    "crisismmd": ["image", "text", "both"],
    "fakeddit": ["image", "text", "both"],
    "fakeddit_real_lcc": ["image", "text", "both"],
    "pheme": ["text"],
}
PS = [0.1, 0.2, 0.4, 0.6, 0.8]
MASK_SEEDS = range(5)
METHOD_SEEDS = (0, 1)

VARIANT_KW = {
    "nfmcd_full": {},
    "nfmcd_all_changes": dict(confidence_mode="consistency", alpha_max=0.6, pca_rank_div=16,
                              single_modality_fill="zero"),
    "nfmcd_rank16": dict(pca_rank_div=16),
}
NONCONST = ["nfmcd_full", "nfmcd_all_changes", "nfmcd_rank16", "spectral_graph+content", "kmeans_content"]
CONST = ["louvain", "nfmcd_structure_only"]
ALL_METHODS = NONCONST + CONST

FIELDS = ["dataset", "mode", "p", "method", "mask_seed", "seed", "k_used", "onmi", "modularity", "f1",
          "frac_nocontent", "frac_no_text", "frac_no_image", "mean_conf", "mean_alpha",
          "fl_both", "fl_text", "fl_image", "fl_none", "fl_unaligned", "error", "secs"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Missing-modality sweep progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_missing.py run --workers 4\n"
        "py run_missing.py summary\n"
        "py run_missing.py verify\n"
        "```\n\n"
        "Results: experiments/missing_results.csv (append-only, fsynced per job). "
        "Summary: experiments/missing_summary.log. Plots: experiments/missing_robustness_*.png.\n"
    ))


def jkey(ds, mode, p, method, mask_seed, seed):
    return (ds, mode, f"{float(p):.2f}", method, int(mask_seed), int(seed))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) != len(FIELDS) or r[0] == "dataset":
                continue
            row = dict(zip(FIELDS, r))
            if row["error"]:
                continue
            try:
                for k in ("p", "onmi", "modularity", "f1", "frac_nocontent", "frac_no_text", "frac_no_image",
                          "mean_conf", "mean_alpha", "fl_both", "fl_text", "fl_image", "fl_none", "fl_unaligned"):
                    row[k] = float(row[k])
                row["mask_seed"] = int(row["mask_seed"])
                row["seed"] = int(row["seed"])
            except ValueError:
                continue
            res[jkey(row["dataset"], row["mode"], row["p"], row["method"], row["mask_seed"], row["seed"])] = row
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


def apply_mask(d, mode, p, mask_seed):
    """Return a copy of dataset dict d with modalities removed (nested in p)."""
    n = d["G"].number_of_nodes()
    rng = np.random.default_rng(1000 + mask_seed)
    u_t = rng.random(n)
    u_v = rng.random(n)
    e_t, e_v = list(d["e_t"]), list(d["e_v"])
    if mode in ("text", "both"):
        e_t = [None if u_t[i] < p else e_t[i] for i in range(n)]
    if mode in ("image", "both"):
        e_v = [None if u_v[i] < p else e_v[i] for i in range(n)]
    d2 = dict(d)
    d2["e_t"], d2["e_v"] = e_t, e_v
    return d2


def spectral_missing(d, seed):
    """Spectral clustering on adjacency + content kNN graph among nodes that
    HAVE content (no-content nodes get graph edges only)."""
    import networkx as nx
    from sklearn.cluster import SpectralClustering
    from nf_mcd import baselines as b

    G = d["G"]
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=None, format="csr").astype(float).toarray()
    X = b.content_matrix(d)
    has = np.linalg.norm(X, axis=1) > 1e-9
    idx = np.where(has)[0]
    if len(idx) > 2:
        Xh = X[idx]
        S = Xh @ Xh.T
        np.fill_diagonal(S, -np.inf)
        kn = min(b.KNN, len(idx) - 1)
        nb = np.argpartition(-S, kn - 1, axis=1)[:, :kn]
        K = np.zeros((n, n))
        rows = np.repeat(idx, kn)
        cols = idx[nb.ravel()]
        K[rows, cols] = 1.0
        K = np.maximum(K, K.T)
        A = A + K
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = SpectralClustering(n_clusters=d["n_communities"], affinity="precomputed",
                                    random_state=seed, assign_labels="kmeans").fit_predict(A)
    return b.Result(labels, b.hard_to_view(labels), d["n_communities"], note="graph+kNN(has-content)")


def run_one(job):
    ds, mode, p, method, mask_seed, seed = job
    t0 = time.monotonic()
    row = dict(dataset=ds, mode=mode, p=f"{p:.2f}", method=method, mask_seed=mask_seed, seed=seed, k_used="",
               onmi=float("nan"), modularity=float("nan"), f1=float("nan"),
               frac_nocontent=float("nan"), frac_no_text=float("nan"), frac_no_image=float("nan"),
               mean_conf=float("nan"), mean_alpha=float("nan"), fl_both=float("nan"), fl_text=float("nan"),
               fl_image=float("nan"), fl_none=float("nan"), fl_unaligned=float("nan"), error="")
    try:
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD

        d0 = load_dataset(ds)
        d = apply_mask(d0, mode, p, mask_seed) if p > 0 else d0
        n = d["G"].number_of_nodes()
        no_t = np.array([v is None for v in d["e_t"]])
        no_v = np.array([v is None for v in d["e_v"]])
        row.update(frac_nocontent=float((no_t & no_v).mean()), frac_no_text=float(no_t.mean()),
                   frac_no_image=float(no_v.mean()))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method in VARIANT_KW:
                model = NFMCD(n_communities=d["n_communities"], seed=seed, **VARIANT_KW[method])
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = b._nfmcd_result(model)
                flags = np.array(model.modality_flags_)
                row.update(mean_conf=float(np.mean(model.confidence_)), mean_alpha=float(np.mean(model.alpha_)),
                           fl_both=float((flags == "both").mean()),
                           fl_text=float((flags == "text_only").mean()),
                           fl_image=float((flags == "image_only").mean()),
                           fl_none=float((flags == "none").mean()),
                           fl_unaligned=float((flags == "both_unaligned").mean()))
            elif method == "spectral_graph+content":
                res = spectral_missing(d, seed)
            elif method == "kmeans_content":
                res = b.kmeans_content(d, seed)
            elif method == "louvain":
                res = b.louvain(d, seed)
            elif method == "nfmcd_structure_only":
                res = b.nfmcd_structure_only(d, seed)
            else:
                raise ValueError(method)
            s = b.score(d, res)
        row.update(k_used=res.k_used, onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    jobs = []
    for ds, modes in DATASET_MODES.items():
        for m in ALL_METHODS:
            for s in METHOD_SEEDS:
                jobs.append((ds, "none", 0.0, m, 0, s))
        for mode in modes:
            for p in PS:
                for ms in MASK_SEEDS:
                    for m in NONCONST:
                        for s in METHOD_SEEDS:
                            jobs.append((ds, mode, p, m, ms, s))
    return jobs


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if jkey(*j) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress(n_done, len(jobs), "running")
    t0 = time.monotonic()
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
                        print(f"  FAILED {row['dataset']}/{row['mode']}/{row['p']}/{row['method']}: {row['error']}",
                              flush=True)
                if n_done % 25 == 0:
                    write_progress(n_done, len(jobs), f"running ({time.monotonic() - t0:.0f}s elapsed)")
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors, {time.monotonic() - t0:.0f}s)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PLIST = [0.0] + PS


def collect(res):
    """curve[(ds, mode, method)][p] -> dict(metric -> list of values)."""
    curve = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in res.values():
        ds, mode, m, p = r["dataset"], r["mode"], r["method"], round(r["p"], 2)
        keys = []
        if mode == "none":
            for md in DATASET_MODES[ds]:
                if m in CONST:
                    keys += [(md, pp) for pp in PLIST]
                else:
                    keys.append((md, 0.0))
        else:
            keys.append((mode, p))
        for md, pp in keys:
            for met in ("onmi", "modularity", "f1"):
                curve[(ds, md, m)][pp][met].append(r[met])
    return curve


def mean_of(curve, ds, mode, m, p, met="onmi"):
    v = curve.get((ds, mode, m), {}).get(round(p, 2), {}).get(met, [])
    return float(np.mean(v)) if v else float("nan")


def sd_of(curve, ds, mode, m, p, met="onmi"):
    v = curve.get((ds, mode, m), {}).get(round(p, 2), {}).get(met, [])
    return float(np.std(v)) if v else float("nan")


def do_summary():
    res = load_results()
    curve = collect(res)
    L = []
    L.append("Missing-modality robustness sweep (MCAR). mean±sd over mask seeds x method seeds (n=10 per cell;"
             " p=0 and constant methods n=2).")
    L.append("k = ground-truth community count; metrics on ALL nodes; LFK ONMI unless stated.")
    L.append("louvain and nfmcd_structure_only ignore content: run once, replicated across p (constant).")
    L.append(f"jobs finished: {len(res)}")

    # No-content share per mode/p
    L.append("\nShare of nodes with NO content (mean over mask seeds; includes nodes already lacking modalities in the cache)")
    for ds, modes in DATASET_MODES.items():
        for mode in modes:
            vals = []
            for p in PLIST:
                if p == 0:
                    rows = [r for r in res.values() if r["dataset"] == ds and r["mode"] == "none" and r["method"] == "nfmcd_full"]
                else:
                    rows = [r for r in res.values() if r["dataset"] == ds and r["mode"] == mode and abs(r["p"] - p) < 1e-9 and r["method"] == "nfmcd_full"]
                vals.append(f"{np.mean([r['frac_nocontent'] for r in rows]):.2f}" if rows else "  - ")
            L.append(f"  {ds:<18}{mode:<6} p=" + " ".join(f"{p:.1f}:{v}" for p, v in zip(PLIST, vals)))

    for ds, modes in DATASET_MODES.items():
        for mode in modes:
            L.append(f"\n=== {ds} | removal mode: {mode} | LFK ONMI vs missing rate p ===")
            L.append(f"{'method':<24}" + "".join(f"{('p=%.1f' % p):>15}" for p in PLIST))
            for m in ALL_METHODS:
                row = f"{m:<24}"
                for p in PLIST:
                    mu = mean_of(curve, ds, mode, m, p)
                    row += f"{'-':>15}" if np.isnan(mu) else f"{('%.3f±%.3f' % (mu, sd_of(curve, ds, mode, m, p))):>15}"
                L.append(row)

    L.append("\n=== Degradation: retention = ONMI(p)/ONMI(0), absolute drop = ONMI(0)-ONMI(p) ===")
    L.append("(retention is meaningless when ONMI(0) is ~0, e.g. PHEME; look at the absolute level there)")
    L.append(f"{'dataset/mode':<28}{'method':<24}{'ONMI(0)':>9}{'ret@0.4':>9}{'drop@0.4':>10}{'ret@0.8':>9}{'drop@0.8':>10}{'ONMI@0.8':>10}")
    robust = {}
    for ds, modes in DATASET_MODES.items():
        for mode in modes:
            for m in NONCONST:
                o0 = mean_of(curve, ds, mode, m, 0.0)
                o4 = mean_of(curve, ds, mode, m, 0.4)
                o8 = mean_of(curve, ds, mode, m, 0.8)
                if np.isnan(o0) or np.isnan(o8):
                    continue
                r4 = o4 / o0 if o0 > 0.02 else float("nan")
                r8 = o8 / o0 if o0 > 0.02 else float("nan")
                L.append(f"{ds + '/' + mode:<28}{m:<24}{o0:>9.3f}{r4:>9.2f}{o0 - o4:>10.3f}{r8:>9.2f}{o0 - o8:>10.3f}{o8:>10.3f}")
                robust[(ds, mode, m)] = (o0, o0 - o8, o8)

    L.append("\n=== Who is most robust? (non-constant methods; smallest absolute drop in ONMI at p=0.8, and best ONMI at p=0.8) ===")
    for ds, modes in DATASET_MODES.items():
        for mode in modes:
            cands = [(robust[(ds, mode, m)][1], m) for m in NONCONST if (ds, mode, m) in robust]
            lvl = [(robust[(ds, mode, m)][2], m) for m in NONCONST if (ds, mode, m) in robust]
            if not cands:
                continue
            cands.sort()
            lvl.sort(reverse=True)
            nf = robust.get((ds, mode, "nfmcd_full"))
            sp = robust.get((ds, mode, "spectral_graph+content"))
            km = robust.get((ds, mode, "kmeans_content"))
            L.append(f"  {ds}/{mode}: smallest drop = {cands[0][1]} ({cands[0][0]:.3f}); best level@0.8 = {lvl[0][1]} ({lvl[0][0]:.3f});"
                     f" drops: NF-MCD {nf[1]:.3f}, spectral {sp[1]:.3f}, kmeans {km[1]:.3f}"
                     if nf and sp and km else f"  {ds}/{mode}: incomplete")

    for met, title in (("modularity", "Modularity"), ("f1", "Membership F1")):
        L.append(f"\n=== {title} at p = 0, 0.4, 0.8 ===")
        L.append(f"{'dataset/mode':<28}{'method':<24}{'p=0':>9}{'p=0.4':>9}{'p=0.8':>9}")
        for ds, modes in DATASET_MODES.items():
            for mode in modes:
                for m in NONCONST:
                    v = [mean_of(curve, ds, mode, m, p, met) for p in (0.0, 0.4, 0.8)]
                    if not any(np.isnan(v)):
                        L.append(f"{ds + '/' + mode:<28}{m:<24}{v[0]:>9.3f}{v[1]:>9.3f}{v[2]:>9.3f}")

    L.append("\n=== How the fuzzy layer reacts (nfmcd_full): mean confidence / mean alpha / node share by modality flag ===")
    L.append("flags: both = paired (CCA space), text/image = single modality, none = no content, unal = both present but CCA not fit")
    L.append(f"{'dataset/mode':<28}{'p':>5}{'conf':>7}{'alpha':>7}{'both':>7}{'text':>7}{'image':>7}{'none':>7}{'unal':>7}")
    for ds, modes in DATASET_MODES.items():
        for mode in modes:
            for p in PLIST:
                if p == 0:
                    rows = [r for r in res.values() if r["dataset"] == ds and r["mode"] == "none" and r["method"] == "nfmcd_full"]
                else:
                    rows = [r for r in res.values() if r["dataset"] == ds and r["mode"] == mode and abs(r["p"] - p) < 1e-9 and r["method"] == "nfmcd_full"]
                if not rows:
                    continue
                g = lambda k: np.mean([r[k] for r in rows])
                L.append(f"{ds + '/' + mode:<28}{p:>5.1f}{g('mean_conf'):>7.2f}{g('mean_alpha'):>7.2f}{g('fl_both'):>7.2f}"
                         f"{g('fl_text'):>7.2f}{g('fl_image'):>7.2f}{g('fl_none'):>7.2f}{g('fl_unaligned'):>7.2f}")

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    make_plots(curve)


def make_plots(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"nfmcd_full": "#0B6E6B", "nfmcd_all_changes": "#5BB5A9", "nfmcd_rank16": "#9CCFC6",
              "spectral_graph+content": "#C2410C", "kmeans_content": "#7C3AED",
              "louvain": "#6B7280", "nfmcd_structure_only": "#A1A1AA"}
    for ds, modes in DATASET_MODES.items():
        fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 3.8), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            for m in ALL_METHODS:
                xs, ys, es = [], [], []
                for p in PLIST:
                    mu = mean_of(curve, ds, mode, m, p)
                    if not np.isnan(mu):
                        xs.append(p); ys.append(mu); es.append(sd_of(curve, ds, mode, m, p))
                if not xs:
                    continue
                ys, es = np.array(ys), np.array(es)
                style = "--" if m in CONST else "-"
                ax.plot(xs, ys, style, color=colors[m], label=m, lw=2 if m == "nfmcd_full" else 1.4, marker="o", ms=3)
                if m not in CONST:
                    ax.fill_between(xs, ys - es, ys + es, color=colors[m], alpha=0.12, lw=0)
            ax.set_title(f"{ds}: remove {mode}")
            ax.set_xlabel("missing rate p")
            ax.set_ylabel("LFK ONMI")
            ax.set_xlim(0, 0.8)
            ax.grid(alpha=0.25)
        axes[0][0].legend(fontsize=7, loc="best")
        fig.tight_layout()
        out = os.path.join(EXP, f"missing_robustness_{ds}.png")
        fig.savefig(out + ".tmp.png", dpi=140)
        plt.close(fig)
        os.replace(out + ".tmp.png", out)


def do_verify():
    res = load_results()
    known = [("crisismmd", "nfmcd_full", 0.609), ("fakeddit", "nfmcd_full", 0.308),
             ("crisismmd", "spectral_graph+content", 0.879)]
    for ds, m, val in known:
        vals = [r["onmi"] for r in res.values() if r["dataset"] == ds and r["method"] == m and r["mode"] == "none"]
        if vals:
            print(f"{ds}/{m}: p=0 mean ONMI {np.mean(vals):.3f} (sd {np.std(vals):.3f}, n={len(vals)}) vs known {val:.3f}")
        else:
            print(f"{ds}/{m}: no p=0 rows yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "verify"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        do_run(args.workers)
        do_summary()
        do_verify()
    elif args.cmd == "summary":
        do_summary()
    else:
        do_verify()
