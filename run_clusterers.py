"""
Compare soft clusterers for NF-MCD Stage 4 against the default fuzzy c-means and baselines.

Usage (from nfmcd_impl/):
    py run_clusterers.py verify              # defaults unchanged (clusterer='fcm' and no argument)
    py run_clusterers.py run [--workers N]   # resumable
    py run_clusterers.py summary             # tables -> experiments/clusterer_summary.log, plot

k = ground-truth community count, seeds 0-2, LFK ONMI. Settings: the five original datasets, the real
Fakeddit graph (full + largest component), the homophily sweep (crisismmd/fakeddit, p_out/p_in),
and the missing-modality "both" mode (p = 0, 0.4, 0.8) for crisismmd and fakeddit_real_lcc.
Crash-safe: each finished job is appended to experiments/clusterer_results.csv (flush + fsync).
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "clusterer_results.csv")
SUMMARY_LOG = os.path.join(EXP, "clusterer_summary.log")
PROGRESS_MD = os.path.join(EXP, "clusterer_progress.md")
VERIFY_TXT = os.path.join(EXP, "clusterer_verify.txt")
PLOT_PNG = os.path.join(EXP, "clusterer_compare.png")

SEEDS = (0, 1, 2)
RATIOS = (0.02, 0.1, 0.25, 0.5, 1.0)
ORIG = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
REAL = ["fakeddit_real_full", "fakeddit_real_lcc"]
SWEEP_C = [f"crisismmd@r{r}" for r in RATIOS]
SWEEP_F = [f"fakeddit@r{r}" for r in RATIOS]
MISS_P = (0.4, 0.8)
MISS_BASES = ("crisismmd", "fakeddit_real_lcc")
MISS = [f"{b}|both@p{p}" for b in MISS_BASES for p in MISS_P]
DATASETS = ORIG + REAL + SWEEP_C + SWEEP_F + MISS

CONFIGS = {
    "default": {},
    "fcm+rawpca+m1.2": dict(content_features="raw_pca", fcm_m=1.2),
    "kmsoft|cca": dict(clusterer="kmeans_softmax"),
    "kmsoft|raw": dict(clusterer="kmeans_softmax", content_features="raw_pca"),
    "gmm|cca": dict(clusterer="gmm"),
    "gmm|raw": dict(clusterer="gmm", content_features="raw_pca"),
    "fcmad|cca": dict(clusterer="fcm_adaptive_m"),
    "fcmad|raw": dict(clusterer="fcm_adaptive_m", content_features="raw_pca"),
    "specsoft|cca": dict(clusterer="spectral_soft"),
    "specsoft|raw": dict(clusterer="spectral_soft", content_features="raw_pca"),
}
BASES = ["base:spectral_graph+content", "base:kmeans_content", "base:louvain"]
METHODS = list(CONFIGS) + BASES
FIELDS = ["method", "dataset", "seed", "k_used", "modularity", "onmi", "f1", "mean_umax", "frac_multi",
          "rule_fidelity", "m_used", "error", "secs"]


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
        "# Clusterer comparison progress\n\n"
        f"- Phase: **{phase}**  ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_clusterers.py run --workers 4\n"
        "py run_clusterers.py summary\n"
        "py run_clusterers.py verify\n"
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
                for k in ("modularity", "onmi", "f1", "mean_umax", "frac_multi", "rule_fidelity", "secs"):
                    row[k] = float(row[k]) if row[k] != "" else float("nan")
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
# data / methods
# ---------------------------------------------------------------------------

def get_d(key, seed):
    from run_realgraph import get_dataset
    if "|both@p" in key:
        from run_missing import apply_mask
        base, p = key.split("|both@p")
        return apply_mask(get_dataset(base, seed), "both", float(p), seed)
    return get_dataset(key, seed)


def has_content(d):
    return any(v is not None for v in d["e_t"]) or any(v is not None for v in d["e_v"])


def fit_config(d, seed, **kw):
    from nf_mcd.pipeline import NFMCD
    model = NFMCD(n_communities=d["n_communities"], seed=seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return model


def run_one(job):
    method, key, seed = job
    t0 = time.monotonic()
    row = dict(method=method, dataset=key, seed=seed, k_used="", modularity=float("nan"), onmi=float("nan"),
               f1=float("nan"), mean_umax="", frac_multi="", rule_fidelity="", m_used="", error="")
    try:
        from nf_mcd import baselines as b
        d = get_d(key, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method in CONFIGS:
                kw = CONFIGS[method]
                model = fit_config(d, seed, **kw)
                res = b._nfmcd_result(model)
                U = model.U_
                row["mean_umax"] = float(U.max(axis=1).mean())
                row["frac_multi"] = float(((U >= b.OVERLAP_THRESHOLD).sum(axis=1) > 1).mean())
                try:
                    row["rule_fidelity"] = float(model.rule_fidelity())
                except Exception:  # noqa: BLE001
                    row["rule_fidelity"] = float("nan")
                row["m_used"] = getattr(model.fcm_result_, "m_used", kw.get("fcm_m", 1.5) if
                                        kw.get("clusterer", "fcm") == "fcm" else "")
            elif method == "base:spectral_graph+content":
                if "|both@p" in key:
                    from run_missing import spectral_missing
                    res = spectral_missing(d, seed)
                else:
                    res = b.spectral_graph_content(d, seed)
            elif method == "base:kmeans_content":
                res = b.kmeans_content(d, seed)
            elif method == "base:louvain":
                res = b.louvain(d, seed)
            else:
                raise ValueError(method)
            s = b.score(d, res)
        row.update(k_used=res.k_used, modularity=s["modularity"], onmi=s["onmi"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    jobs = []
    for ds in DATASETS:
        d = get_d(ds, 0)
        content = has_content(d)
        for m in METHODS:
            if m == "base:kmeans_content" and not content:
                continue
            jobs += [(m, ds, s) for s in SEEDS]
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
                if n_done % 25 == 0:
                    write_progress("run", n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress("run", n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# verify: defaults unchanged
# ---------------------------------------------------------------------------

def _read_csv(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            out[(row["method"], row["dataset"], int(row["seed"]))] = row
    return out


def do_verify():
    from nf_mcd import baselines as b
    from run_realgraph import get_dataset
    base = _read_csv(os.path.join(EXP, "baselines_results.csv"))
    rg = _read_csv(os.path.join(EXP, "realgraph_results.csv"))
    checks = []
    for ds in ORIG:
        for s in SEEDS:
            checks.append((ds, s, base[("nfmcd_full", ds, s)], "baselines_results.csv"))
    for ds in REAL + SWEEP_C + SWEEP_F:
        for s in SEEDS:
            checks.append((ds, s, rg[("var:default", ds, s)], "realgraph_results.csv"))
    worst, bad = 0.0, 0
    for ds, s, ref, src in checks:
        d = get_dataset(ds, s)
        for variant in ("no-arg", "clusterer=fcm"):
            kw = {} if variant == "no-arg" else dict(clusterer="fcm")
            model = fit_config(d, s, **kw)
            sc = b.score(d, b._nfmcd_result(model))
            diffs = [abs(sc[m] - float(ref[m])) for m in ("modularity", "onmi", "f1")]
            worst = max(worst, max(diffs))
            if max(diffs) > 1e-9:
                bad += 1
                print(f"MISMATCH {ds} seed {s} ({variant}) vs {src}: diffs={diffs}")
    msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} verified {len(checks)} saved default rows x 2 call styles; "
           f"mismatches={bad}; max abs diff={worst:.3e}\n")
    print(msg, end="")
    atomic_write_text(VERIFY_TXT, msg)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _agg(res):
    cell = defaultdict(list)
    for (m, ds, s), r in res.items():
        cell[(m, ds)].append(r)
    out = {}
    for key, rows in cell.items():
        rec = {}
        for fld in ("onmi", "modularity", "f1", "mean_umax", "frac_multi", "rule_fidelity", "secs"):
            v = np.array([r[fld] for r in rows], dtype=float)
            rec[fld] = (float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan"),
                        float(np.nanstd(v)) if not np.all(np.isnan(v)) else float("nan"))
        rec["n"] = len(rows)
        out[key] = rec
    return out


def _f(v, w=6):
    return f"{v:.3f}".rjust(w) if v == v else "n/a".rjust(w)


def _table(agg, methods, datasets, field, heads, title):
    lines = [title, "-" * len(title)]
    name_w = max(len(m) for m in methods) + 2
    lines.append("method".ljust(name_w) + "".join(h.rjust(13) for h in heads))
    for m in methods:
        cells = []
        for ds in datasets:
            rec = agg.get((m, ds))
            if rec is None:
                cells.append("n/a".rjust(13))
            else:
                mu, sd = rec[field]
                cells.append((f"{mu:.3f}±{sd:.3f}" if mu == mu else "n/a").rjust(13))
        lines.append(m.ljust(name_w) + "".join(cells))
    lines.append("")
    return lines


def _wtl(agg, a, b):
    w = t = l = 0
    detail = []
    for ds in DATASETS:
        ra, rb = agg.get((a, ds)), agg.get((b, ds))
        if ra is None or rb is None:
            continue
        (ma, sa), (mb, sb) = ra["onmi"], rb["onmi"]
        diff = ma - mb
        tol = max(0.02, sa + sb)
        if abs(diff) < tol:
            t += 1
        elif diff > 0:
            w += 1
        else:
            l += 1
        detail.append((ds, diff))
    return w, t, l


def do_summary():
    res = load_results()
    if not res:
        print("no results yet")
        return
    agg = _agg(res)
    nf = list(CONFIGS)
    allm = nf + BASES
    L = ["Soft clusterers for NF-MCD Stage 4 (mean±sd over seeds 0,1,2; k = ground-truth community count)", ""]
    L.append("Fixed a-priori hyperparameters (never tuned per dataset): softmax temperature tau = 0.5 * median")
    L.append("squared distance to nearest k-means center; fcm_adaptive_m ladder (1.5,1.3,1.2,1.1) accepting the")
    L.append("largest m with mean(U.max) >= 1/k + 0.25*(1-1/k); spectral_soft: 12-NN graph + k Laplacian")
    L.append("eigenvectors + kmeans_softmax; gmm: spherical covariance, n_init=3, reg_covar=1e-4.")
    L.append("Names: |cca = CCA fused content (default), |raw = raw-PCA content (content_features='raw_pca').")
    L.append("")

    main = ORIG + REAL
    heads = ["CrisisMMD", "PHEME", "Fakeddit", "DBLP", "Amazon", "FakedReal", "FakedRealLCC"]
    L += _table(agg, allm, main, "onmi", heads, "Table 1: LFK ONMI, original datasets + real Fakeddit graph")
    L += _table(agg, allm, main, "modularity", heads, "Table 1b: modularity of the hard partition")
    L += _table(agg, allm, main, "f1", heads, "Table 1c: membership F1")
    rh = [f"r={r}" for r in RATIOS]
    L += _table(agg, allm, SWEEP_C, "onmi", rh, "Table 2a: CrisisMMD homophily sweep, LFK ONMI (r = p_out/p_in; 1.0 = no label info)")
    L += _table(agg, allm, SWEEP_F, "onmi", rh, "Table 2b: Fakeddit homophily sweep, LFK ONMI")
    mh = ["CrisMMD p=0", "p=0.4", "p=0.8", "RealLCC p=0", "p=0.4", "p=0.8"]
    mcols = ["crisismmd", "crisismmd|both@p0.4", "crisismmd|both@p0.8",
             "fakeddit_real_lcc", "fakeddit_real_lcc|both@p0.4", "fakeddit_real_lcc|both@p0.8"]
    L += _table(agg, allm, mcols, "onmi", mh, "Table 3: missing-modality 'both' mode, LFK ONMI (p = share of each modality removed)")

    # membership quality
    L.append("Table 4: membership quality of the soft assignments (mean over all settings where the config ran)")
    L.append("-" * 96)
    L.append("config".ljust(20) + "mean U.max".rjust(12) + "multi>=0.2".rjust(12) + "rule_fid".rjust(10)
             + "U.max@hard*".rjust(13) + "multi@hard*".rjust(13) + "secs/fit".rjust(10))
    hard_ds = ["crisismmd@r0.5", "crisismmd@r1.0", "fakeddit@r0.5", "fakeddit@r1.0"]
    for m in nf:
        def avg(fld, dss):
            v = [agg[(m, d)][fld][0] for d in dss if (m, d) in agg and agg[(m, d)][fld][0] == agg[(m, d)][fld][0]]
            return float(np.mean(v)) if v else float("nan")
        L.append(m.ljust(20) + _f(avg("mean_umax", DATASETS), 12) + _f(avg("frac_multi", DATASETS), 12)
                 + _f(avg("rule_fidelity", DATASETS), 10) + _f(avg("mean_umax", hard_ds), 13)
                 + _f(avg("frac_multi", hard_ds), 13) + _f(avg("secs", DATASETS), 10))
    L.append("* hard = sweep settings with no label-informative structure (r = 0.5, 1.0 for both datasets); a mean U.max")
    L.append("  near 1/k (0.14 for k=7, 0.10 for k=10) means the uniform-membership collapse.")
    L.append("")

    # W/T/L
    L.append("Win / tie / loss of each config, by ONMI over all settings (tie if |diff| < max(0.02, sd_a + sd_b))")
    L.append("-" * 96)
    L.append("config".ljust(20) + "vs default".rjust(16) + "vs spectral+content".rjust(24) + "vs kmeans_content".rjust(22))
    for m in nf:
        cells = []
        for other in ("default", "base:spectral_graph+content", "base:kmeans_content"):
            if m == other:
                cells.append("-")
            else:
                w, t, l = _wtl(agg, m, other)
                cells.append(f"{w}W/{t}T/{l}L")
        L.append(m.ljust(20) + cells[0].rjust(16) + cells[1].rjust(24) + cells[2].rjust(22))
    w, t, l = _wtl(agg, "base:spectral_graph+content", "default")
    L.append("(reference) spectral+content vs default: " + f"{w}W/{t}T/{l}L")
    L.append("")

    # mean over settings
    L.append("Mean ONMI over all settings and over groups")
    L.append("-" * 96)
    groups = {"orig5": ORIG, "real": REAL, "sweep": SWEEP_C + SWEEP_F, "missing": MISS + ["crisismmd", "fakeddit_real_lcc"],
              "all": DATASETS}
    L.append("config".ljust(24) + "".join(g.rjust(10) for g in groups))
    for m in allm:
        cells = []
        for g, dss in groups.items():
            v = [agg[(m, d)]["onmi"][0] for d in dss if (m, d) in agg]
            cells.append(_f(float(np.mean(v)) if v else float("nan"), 10))
        L.append(m.ljust(24) + "".join(cells))
    L.append("(missing group includes both p=0 columns and p in {0.4, 0.8}; the 'all' mean skips the "
             "content-only baseline where it does not apply.)")
    L.append("")
    atomic_write_text(SUMMARY_LOG, "\n".join(L) + "\n")
    print("\n".join(L))
    do_plot(agg)


def do_plot(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["default", "fcm+rawpca+m1.2", "kmsoft|raw", "gmm|raw", "fcmad|raw", "specsoft|raw",
             "base:spectral_graph+content", "base:kmeans_content"]
    colors = {"default": "#888888", "fcm+rawpca+m1.2": "#6a3d9a", "kmsoft|raw": "#1f78b4", "gmm|raw": "#33a02c",
              "fcmad|raw": "#ff7f00", "specsoft|raw": "#e31a1c", "base:spectral_graph+content": "#000000",
              "base:kmeans_content": "#b15928"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    def series(m, keys):
        return [agg[(m, k)]["onmi"][0] if (m, k) in agg else np.nan for k in keys]

    panels = [
        ("CrisisMMD sweep", SWEEP_C, [str(r) for r in RATIOS], "p_out / p_in"),
        ("Fakeddit sweep", SWEEP_F, [str(r) for r in RATIOS], "p_out / p_in"),
        ("CrisisMMD missing (both)", ["crisismmd", "crisismmd|both@p0.4", "crisismmd|both@p0.8"], ["0", "0.4", "0.8"], "share missing"),
        ("Real Fakeddit LCC missing (both)", ["fakeddit_real_lcc", "fakeddit_real_lcc|both@p0.4", "fakeddit_real_lcc|both@p0.8"],
         ["0", "0.4", "0.8"], "share missing"),
    ]
    for ax, (title, keys, xt, xl) in zip(axes, panels):
        for m in lines:
            ax.plot(range(len(keys)), series(m, keys), marker="o", ms=4, color=colors[m], label=m,
                    lw=2 if m in ("default", "base:spectral_graph+content") else 1.3,
                    ls="--" if m.startswith("base:") else "-")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(xt)
        ax.set_xlabel(xl)
        ax.set_ylabel("LFK ONMI")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "verify":
        do_verify()
    elif args.cmd == "run":
        do_run(args.workers)
    else:
        do_summary()


if __name__ == "__main__":
    main()
