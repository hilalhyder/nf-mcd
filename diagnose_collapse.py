"""
Why does NF-MCD extract no content signal when structure is uninformative?

Usage (from nfmcd_impl/):
    py diagnose_collapse.py verify              # defaults-unchanged check vs saved results
    py diagnose_collapse.py run [--workers N]   # diagnostics + ceilings + remedies (resumable)
    py diagnose_collapse.py summary             # tables -> experiments/collapse_summary.log, plot

Method families (all k = ground-truth community count, seeds 0-2, LFK ONMI):
  diag                 default NFMCD internals (cluster sizes, U.max, block variances, eta^2, ...)
  var:default          NFMCD defaults
  rem:*                optional remedies (new NFMCD parameters, defaults unchanged)
  ceil:*               fuzzy c-means (m=1.5) on a single feature block, no fusion
  base:*               reference baselines from nf_mcd.baselines
Crash-safe: each finished job is appended to experiments/collapse_results.csv (flush+fsync).
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
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "collapse_results.csv")
SUMMARY_LOG = os.path.join(EXP, "collapse_summary.log")
PROGRESS_MD = os.path.join(EXP, "collapse_progress.md")
PLOT_PNG = os.path.join(EXP, "collapse_remedies.png")

SEEDS = (0, 1, 2)
RATIOS = (0.02, 0.1, 0.25, 0.5, 1.0)
BASES = ("crisismmd", "fakeddit")
SWEEP = [f"{b}@r{r}" for b in BASES for r in RATIOS]
OTHER = ["crisismmd", "fakeddit", "fakeddit_real_full", "fakeddit_real_lcc", "pheme", "dblp", "amazon"]
DATASETS = SWEEP + OTHER
DIAG_DATASETS = SWEEP + ["crisismmd", "fakeddit"]

REMEDIES = {
    "var:default": {},
    "rem:raw_pca": dict(content_features="raw_pca"),
    "rem:std_blocks": dict(standardize_blocks=True),
    "rem:raw_pca+std": dict(content_features="raw_pca", standardize_blocks=True),
    "rem:m1.2": dict(fcm_m=1.2),
    "rem:raw_pca+m1.2": dict(content_features="raw_pca", fcm_m=1.2),
    "rem:raw_pca+std+m1.2": dict(content_features="raw_pca", standardize_blocks=True, fcm_m=1.2),
}
CEILINGS = ["ceil:content_cca", "ceil:content_cca_weighted", "ceil:raw_text", "ceil:raw_image",
            "ceil:raw_concat_pca16", "ceil:structure"]
BASELINES = ["base:louvain", "base:spectral_graph+content", "base:kmeans_content", "base:nfmcd_structure_only"]
# Attribution ceilings: same feature blocks, different clusterer (k-means) or lower FCM m.
ATTRIB = ["ceil:fused_m1.1", "ceil:fused_m1.2", "kceil:fused", "kceil:content_cca", "kceil:raw_pca16", "kceil:structure"]
METHODS = list(REMEDIES) + CEILINGS + ATTRIB + BASELINES
ATTRIB_DATASETS = set(SWEEP + ["crisismmd", "fakeddit"])
FIELDS = ["method", "dataset", "seed", "k_used", "modularity", "onmi", "onmi_hard", "ari", "f1", "extra", "error", "secs"]


# ---------------------------------------------------------------------------
# io helpers
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
        "# Collapse diagnosis progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py diagnose_collapse.py run --workers 4\n"
        "py diagnose_collapse.py summary\n"
        "py diagnose_collapse.py verify\n"
        "```\n"
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
                for k in ("modularity", "onmi", "onmi_hard", "ari", "f1"):
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


# ---------------------------------------------------------------------------
# model / scoring helpers
# ---------------------------------------------------------------------------

def has_content(d):
    return any(v is not None for v in d["e_t"]) or any(v is not None for v in d["e_v"])


def fit_model(d, seed, **kw):
    from nf_mcd.pipeline import NFMCD
    model = NFMCD(n_communities=d["n_communities"], seed=seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return model


def primary_labels(d):
    return np.array([min(t) for t in d["true"]])


def full_score(d, res):
    from sklearn.metrics import adjusted_rand_score
    from nf_mcd import baselines as b
    from nf_mcd import metrics as mx
    s = b.score(d, res)
    true_view = b._true_view(d["true"], d["n_communities"])
    n = d["G"].number_of_nodes()
    s["onmi_hard"] = float(mx.overlapping_nmi(b.hard_to_view(res.hard), true_view, n))
    s["ari"] = float(adjusted_rand_score(primary_labels(d), res.hard))
    return s


def fcm_result(X, k, seed):
    from nf_mcd import baselines as b
    from nf_mcd import community_detection as cd
    U = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(X).U
    return b.Result(cd.defuzzify(U), b._node_to_comm_view(U), k)


def raw_block(vecs, n):
    dim = next((v.shape[0] for v in vecs if v is not None), None)
    if dim is None:
        return None
    M = np.zeros((n, dim))
    for i, v in enumerate(vecs):
        if v is not None:
            M[i] = v / (np.linalg.norm(v) + 1e-12)
    return M


def eta2(X, lab):
    mu = X.mean(axis=0)
    tot = float(np.sum((X - mu) ** 2))
    if tot <= 1e-12:
        return float("nan")
    bet = 0.0
    for g in np.unique(lab):
        m = lab == g
        bet += m.sum() * float(np.sum((X[m].mean(axis=0) - mu) ** 2))
    return bet / tot


def pair_dist(X, seed=0, m=300):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(m, X.shape[0]), replace=False)
    Y = X[idx]
    D = np.sqrt(np.maximum(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1), 0))
    iu = np.triu_indices(len(idx), 1)
    return float(D[iu].mean())


def diagnose(d, seed):
    from nf_mcd import topology as topo
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    model = fit_model(d, seed)
    U = model.U_
    hard = np.argmax(U, axis=1)
    lab = primary_labels(d)
    sizes = np.sort(np.bincount(hard, minlength=k))[::-1]
    umax = U.max(axis=1)
    node_view = model.overlapping_communities(0.2)
    n_per_node = np.array([len(s) for s in node_view])
    pview = model.overlapping_communities_view(0.2)
    psz = np.sort(np.array([len(c) for c in pview]))[::-1] / n
    centers = model.fcm_result_.centers
    Xf = topo.fuse_features(model.fused_content_, model.Z_s_, model.alpha_)
    within = float(np.sqrt(np.mean(np.sum((Xf - centers[hard]) ** 2, axis=1))))
    cd_ = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(k, 1)
    a = model.alpha_.reshape(-1, 1)
    Zc_w = np.sqrt(a) * model.fused_content_
    Zs_w = np.sqrt(1 - a) * model.Z_s_

    def tv(Z):
        return float(np.mean(np.sum((Z - Z.mean(axis=0, keepdims=True)) ** 2, axis=1)))

    vc, vs = tv(Zc_w), tv(Zs_w)
    return dict(
        n_nodes=n, k=k,
        max_cluster_frac=float(sizes[0] / n), min_cluster_frac=float(sizes[-1] / n),
        n_empty=int((sizes == 0).sum()), sizes_top3=[int(x) for x in sizes[:3]],
        umax_mean=float(umax.mean()), umax_q10=float(np.quantile(umax, 0.1)),
        umax_q90=float(np.quantile(umax, 0.9)), frac_umax_lt_0p3=float((umax < 0.3).mean()),
        one_over_k=1.0 / k,
        comms_per_node=float(n_per_node.mean()), frac_in_half_or_more=float((n_per_node >= k / 2).mean()),
        pred_comm_frac_top=float(psz[0]), pred_comm_frac_med=float(np.median(psz)),
        fcm_obj=float(model.fcm_result_.objective_history[-1]), fcm_iters=int(model.fcm_result_.n_iter),
        center_min_dist=float(cd_[iu].min()), center_mean_dist=float(cd_[iu].mean()), within_rms=within,
        center_sep_ratio=float(cd_[iu].min() / (within + 1e-12)),
        alpha_mean=float(model.alpha_.mean()), conf_mean=float(model.confidence_.mean()),
        frac_both=float(np.mean([f == "both" for f in model.modality_flags_])),
        var_content_w=vc, var_struct_w=vs, var_share_content=vc / (vc + vs + 1e-12),
        pdist_content_w=pair_dist(Zc_w, seed), pdist_struct_w=pair_dist(Zs_w, seed),
        cca_dim=int(model.fused_content_.shape[1]),
        eta2_content_w=eta2(Zc_w, lab), eta2_struct_w=eta2(Zs_w, lab), eta2_fused=eta2(Xf, lab),
        eta2_content_raw=eta2(model.fused_content_, lab), eta2_struct_raw=eta2(model.Z_s_, lab),
    )


# ---------------------------------------------------------------------------
# job execution
# ---------------------------------------------------------------------------

def applicable(method, d, key=None):
    if method.startswith("diag"):
        return True
    if method in ATTRIB:
        return key in ATTRIB_DATASETS and (has_content(d) or method == "kceil:structure")
    if method in ("ceil:raw_image",):
        return any(v is not None for v in d["e_v"])
    if method in ("ceil:raw_text",):
        return any(v is not None for v in d["e_t"])
    if method.startswith("ceil:") and method != "ceil:structure":
        return has_content(d)
    if method == "base:kmeans_content":
        return has_content(d)
    return True


def run_method(method, d, seed):
    from nf_mcd import baselines as b
    from nf_mcd import topology as topo
    k = d["n_communities"]
    n = d["G"].number_of_nodes()
    if method in REMEDIES:
        return b._nfmcd_result(fit_model(d, seed, **REMEDIES[method]))
    if method in ATTRIB:
        from sklearn.cluster import KMeans
        from nf_mcd import community_detection as cd
        name = method.split(":", 1)[1]
        if name.startswith("fused"):
            model = fit_model(d, seed)
            X = topo.fuse_features(model.fused_content_, model.Z_s_, model.alpha_)
        elif name == "content_cca":
            X = fit_model(d, seed).fused_content_
        elif name == "raw_pca16":
            X = topo.raw_pca_content(d["e_t"], d["e_v"], dim=16, seed=seed)
        elif name == "structure":
            X = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
        else:
            raise ValueError(method)
        if method.startswith("ceil:fused_m"):
            m = float(name.split("_m")[1])
            U = cd.FuzzyCMeans(n_clusters=k, m=m, seed=seed).fit(X).U
            return b.Result(cd.defuzzify(U), b._node_to_comm_view(U), k)
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        return b.Result(labels, b.hard_to_view(labels), k)
    if method.startswith("ceil:"):
        name = method[5:]
        if name == "structure":
            X = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
        elif name in ("content_cca", "content_cca_weighted"):
            model = fit_model(d, seed)
            X = model.fused_content_
            if name.endswith("weighted"):
                X = np.sqrt(model.alpha_.reshape(-1, 1)) * X
        elif name == "raw_text":
            X = raw_block(d["e_t"], n)
        elif name == "raw_image":
            X = raw_block(d["e_v"], n)
        elif name == "raw_concat_pca16":
            X = topo.raw_pca_content(d["e_t"], d["e_v"], dim=16, seed=seed)
        else:
            raise ValueError(method)
        return fcm_result(X, k, seed)
    name = method[5:]
    if name == "nfmcd_structure_only":
        return b.nfmcd_structure_only(d, seed)
    return b.METHODS[name][0](d, seed)


def run_one(job):
    from run_realgraph import get_dataset
    method, key, seed = job
    t0 = time.monotonic()
    row = dict(method=method, dataset=key, seed=seed, k_used="", modularity=float("nan"), onmi=float("nan"),
               onmi_hard=float("nan"), ari=float("nan"), f1=float("nan"), extra="", error="")
    try:
        d = get_dataset(key, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "diag":
                row["extra"] = json.dumps(diagnose(d, seed))
                row["k_used"] = d["n_communities"]
            else:
                res = run_method(method, d, seed)
                s = full_score(d, res)
                row.update(k_used=res.k_used, modularity=s["modularity"], onmi=s["onmi"],
                           onmi_hard=s["onmi_hard"], ari=s["ari"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    from run_realgraph import get_dataset
    jobs = []
    for ds in DATASETS:
        d = get_dataset(ds, 0)
        for m in METHODS:
            if applicable(m, d, ds):
                jobs += [(m, ds, s) for s in SEEDS]
        if ds in DIAG_DATASETS:
            jobs += [("diag", ds, s) for s in SEEDS]
    return jobs


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if j not in done]
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
                    if n_err <= 5:
                        print(f"  FAILED {row['method']}/{row['dataset']}/{row['seed']}: {row['error']}", flush=True)
                if n_done % 20 == 0:
                    write_progress("run", n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress("run", n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# verify: defaults unchanged
# ---------------------------------------------------------------------------

def _read_csv(path, method_col="method"):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("error"):
                continue
            out[(row["method"], row["dataset"], int(row["seed"]))] = row
    return out


def do_verify():
    from run_realgraph import get_dataset
    base = _read_csv(os.path.join(EXP, "baselines_results.csv"))
    rg = _read_csv(os.path.join(EXP, "realgraph_results.csv"))
    checks = []
    for ds in ("crisismmd", "pheme", "fakeddit", "dblp", "amazon"):
        for s in SEEDS:
            checks.append(("nfmcd_full", ds, s, base[("nfmcd_full", ds, s)], "baselines_results.csv"))
    for ds in ["fakeddit_real_full", "fakeddit_real_lcc"] + SWEEP:
        for s in SEEDS:
            checks.append(("var:default", ds, s, rg[("var:default", ds, s)], "realgraph_results.csv"))
    worst = 0.0
    bad = 0
    for method, ds, s, ref, src in checks:
        d = get_dataset(ds, s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_method("var:default", d, s)
            sc = full_score(d, res)
        diffs = [abs(sc[m] - float(ref[m])) for m in ("modularity", "onmi", "f1")]
        worst = max(worst, max(diffs))
        if max(diffs) > 1e-9:
            bad += 1
            print(f"MISMATCH {ds} seed {s} vs {src}: diffs={diffs}")
    print(f"verified {len(checks)} default rows against saved results; mismatches={bad}; max abs diff={worst:.2e}")
    atomic_write_text(os.path.join(EXP, "collapse_verify.txt"),
                      f"{time.strftime('%Y-%m-%d %H:%M:%S')} verified {len(checks)} rows; mismatches={bad}; max abs diff={worst:.3e}\n")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _ms(v):
    a = np.array(v, dtype=float)
    return f"{np.nanmean(a):.3f}+-{np.nanstd(a):.3f}"


def do_summary():
    res = load_results()
    agg = defaultdict(lambda: defaultdict(list))
    diag = defaultdict(list)
    for (m, ds, s), r in res.items():
        if m == "diag":
            diag[ds].append(json.loads(r["extra"]))
            continue
        for k in ("modularity", "onmi", "onmi_hard", "ari", "f1"):
            agg[(m, ds)][k].append(r[k])
    L = []

    def dmean(ds, key):
        v = [x[key] for x in diag[ds]]
        return float(np.mean(v)) if v else float("nan")

    L.append("1. What the default model outputs (mean over 3 seeds; k = ground-truth count)")
    L.append("-" * 118)
    cols = [("max_cluster_frac", "maxClu"), ("min_cluster_frac", "minClu"), ("umax_mean", "U.max"), ("frac_umax_lt_0p3", "U<.3"),
            ("comms_per_node", "comm/n"), ("pred_comm_frac_med", "predSz"), ("center_sep_ratio", "sep"), ("alpha_mean", "alpha")]
    L.append(f"{'dataset':<20}" + "".join(f"{c[1]:>9}" for c in cols) + f"{'ONMI':>9}{'ONMIhard':>10}{'ARI':>8}")
    for ds in DIAG_DATASETS:
        if not diag[ds]:
            continue
        L.append(f"{ds:<20}" + "".join(f"{dmean(ds, c[0]):>9.3f}" for c in cols)
                 + f"{np.mean(agg[('var:default', ds)]['onmi']):>9.3f}{np.mean(agg[('var:default', ds)]['onmi_hard']):>10.3f}"
                 + f"{np.mean(agg[('var:default', ds)]['ari']):>8.3f}")
    L.append("  maxClu/minClu = largest/smallest hard cluster fraction; U.max = mean top membership (1/k is uniform); "
             "U<.3 = fraction of nodes with top membership <0.3;")
    L.append("  comm/n = mean communities per node in the overlap view (threshold 0.2 + argmax); predSz = median predicted community size / n;")
    L.append("  sep = min centre distance / within-cluster RMS; alpha = mean content weight.")
    L.append("")

    L.append("2. Which block carries the signal (means over seeds)")
    L.append("-" * 118)
    cols2 = [("var_share_content", "varShare_c"), ("pdist_content_w", "pdist_c"), ("pdist_struct_w", "pdist_s"),
             ("eta2_content_w", "eta2_c_w"), ("eta2_struct_w", "eta2_s_w"), ("eta2_fused", "eta2_fused"),
             ("eta2_content_raw", "eta2_c_raw"), ("cca_dim", "cca_dim")]
    L.append(f"{'dataset':<20}" + "".join(f"{c[1]:>12}" for c in cols2))
    for ds in DIAG_DATASETS:
        if diag[ds]:
            L.append(f"{ds:<20}" + "".join(f"{dmean(ds, c[0]):>12.3f}" for c in cols2))
    L.append("  varShare_c = variance share of the (alpha-weighted) content block in the fused vector; pdist = mean pairwise distance in each weighted block;")
    L.append("  eta2 = between-true-community variance fraction of the block (label signal available to a clustering of that block).")
    L.append("")

    def table(metric, datasets, title, methods):
        L.append(title)
        L.append("-" * len(title))
        L.append(f"{'method':<30}" + "".join(f"{d:>21}" for d in datasets))
        for m in methods:
            row = f"{m:<30}"
            for ds in datasets:
                row += f"{_ms(agg[(m, ds)][metric]):>21}" if (m, ds) in agg else f"{'n/a':>21}"
            L.append(row)
        L.append("")

    focus = [f"{b}@r{r}" for b in BASES for r in (0.25, 0.5, 1.0)]
    table("onmi", focus, "3. Single-block ceilings (FCM m=1.5, same k) vs fused default, LFK ONMI (overlap view)", CEILINGS + ["var:default"])
    table("onmi_hard", focus, "3b. Same, ONMI of the HARD partition only (separates overlap thresholding from partition quality)", CEILINGS + ["var:default"])
    table("onmi", focus, "3c. Attribution: same fused features / blocks with k-means or lower FCM m (kceil = k-means, hard partition), LFK ONMI",
          ["var:default"] + ATTRIB + ["base:kmeans_content"])
    table("ari", focus, "3d. Same, adjusted Rand index of the hard partition vs primary label",
          ["var:default"] + ATTRIB + ["base:kmeans_content"])
    rem_methods = list(REMEDIES) + ["base:spectral_graph+content", "base:kmeans_content", "base:nfmcd_structure_only", "base:louvain"]
    for b in BASES:
        table("onmi", [f"{b}@r{r}" for r in RATIOS], f"4. Remedies vs baselines, {b} SBM sweep: LFK ONMI by p_out/p_in", rem_methods)
    other = ["crisismmd", "fakeddit", "fakeddit_real_full", "fakeddit_real_lcc", "pheme", "dblp", "amazon"]
    table("onmi", other, "5. Remedies on leaky / real graphs, PHEME, DBLP, Amazon: LFK ONMI", rem_methods)
    table("modularity", other, "5b. Same, modularity", rem_methods)
    table("f1", other, "5c. Same, membership F1", rem_methods)
    table("modularity", [f"{b}@r{r}" for b in BASES for r in (0.02, 0.5, 1.0)], "5d. Sweep modularity (selected ratios)", rem_methods)

    L.append("6. Remedy minus default (ONMI, mean over seeds; positive = better)")
    L.append("-" * 70)
    all_ds = SWEEP + other
    L.append(f"{'dataset':<22}" + "".join(f"{m.split(':')[1]:>16}" for m in list(REMEDIES)[1:]))
    for ds in all_ds:
        if ("var:default", ds) not in agg:
            continue
        base = np.mean(agg[("var:default", ds)]["onmi"])
        L.append(f"{ds:<22}" + "".join(
            f"{np.mean(agg[(m, ds)]['onmi']) - base:>+16.3f}" if (m, ds) in agg else f"{'n/a':>16}" for m in list(REMEDIES)[1:]))
    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    do_plot(agg)


def do_plot(agg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if agg is None:
        res = load_results()
        agg = defaultdict(lambda: defaultdict(list))
        for (m, ds, s), r in res.items():
            if m != "diag":
                agg[(m, ds)]["onmi"].append(r["onmi"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    lines = list(REMEDIES) + ["base:spectral_graph+content", "base:kmeans_content", "base:nfmcd_structure_only",
                              "ceil:raw_concat_pca16"]
    for ax, b in zip(axes, BASES):
        for m in lines:
            ys = [np.mean(agg[(m, f"{b}@r{r}")]["onmi"]) if (m, f"{b}@r{r}") in agg else np.nan for r in RATIOS]
            ax.plot(RATIOS, ys, marker="o", label=m, linewidth=2.2 if m == "var:default" else 1.2)
        ax.set_xscale("log")
        ax.set_xlabel("p_out / p_in  (1.0 = structure has no label information)")
        ax.set_title(f"{b}: LFK ONMI, default vs remedies vs baselines (synthetic SBM)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LFK ONMI")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_PNG + ".tmp.png", dpi=140)
    os.replace(PLOT_PNG + ".tmp.png", PLOT_PNG)
    plt.close(fig)
    print("saved", PLOT_PNG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "verify":
        do_verify()
    elif a.cmd == "run":
        do_run(a.workers)
    else:
        do_summary()
