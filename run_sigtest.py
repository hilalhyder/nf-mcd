"""
Statistical backing for this project's headline head-to-head comparisons.

Every earlier comparison in this project (experiments/baselines_summary.log,
robust_refresh_summary.log, clusterer_summary.log, ...) used 3 seeds and a
tie rule of |diff| < max(0.02, sd_a+sd_b) -- a heuristic, not a real test,
and with n=3 essentially powerless. This script reruns three representative
comparisons at n=20 seeds with a real paired t-test, Wilcoxon signed-rank,
bootstrap CIs, and a retrospective power analysis of the n=3 regime.

Datasets: crisismmd (multimodal, leaky/sampled graph), fakeddit (multimodal,
leaky/sampled graph), dblp (content-free, real SNAP graph).
Methods: nfmcd_default, nfmcd_robust, spectral_graph+content, kmeans_content
(skipped on dblp -- no content).
Metric: LFK overlapping_nmi (primary), modularity + F1 reported too.

Read-only reuse: nf_mcd/baselines.py (score, spectral_graph_content,
kmeans_content, _nfmcd_result), nf_mcd/pipeline.py (NFMCD, NFMCD.robust),
run_realgraph.py's cached dataset pickles. Nothing in nf_mcd/* is modified.

Usage:
    py run_sigtest.py run [--workers N]
    py run_sigtest.py summary
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "sigtest_results.csv")
SUMMARY_LOG = os.path.join(EXP, "sigtest_summary.log")
PROGRESS_MD = os.path.join(EXP, "sigtest_progress.md")
CI_PNG = os.path.join(EXP, "sigtest_ci.png")

DATASETS = ["crisismmd", "fakeddit", "dblp"]
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
METHODS_ALL = ["nfmcd_default", "nfmcd_robust", "spectral_graph+content", "kmeans_content"]
FIELDS = ["method", "dataset", "seed", "k_used", "modularity", "onmi", "f1", "error", "secs"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(phase, done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Significance-test progress\n\n"
        f"- Phase: **{phase}** ({done}/{total})\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_sigtest.py run --workers 4\n"
        "py run_sigtest.py summary\n"
        "```\n"
    ))


def methods_for(dataset):
    if dataset == "dblp":
        return [m for m in METHODS_ALL if m != "kmeans_content"]
    return list(METHODS_ALL)


def all_jobs():
    return [(m, ds, s) for ds in DATASETS for m in methods_for(ds) for s in SEEDS]


def load_dataset(key):
    with open(os.path.join(CACHE, f"{key}.pkl"), "rb") as f:
        return pickle.load(f)


def run_one(job):
    method, dataset, seed = job
    t0 = time.monotonic()
    row = dict(method=method, dataset=dataset, seed=seed, k_used="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"), error="")
    try:
        from nf_mcd import baselines as b
        from nf_mcd.pipeline import NFMCD
        d = load_dataset(dataset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "nfmcd_default":
                model = NFMCD(n_communities=d["n_communities"], seed=seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = b._nfmcd_result(model)
            elif method == "nfmcd_robust":
                model = NFMCD.robust(n_communities=d["n_communities"], seed=seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = b._nfmcd_result(model)
            elif method == "spectral_graph+content":
                res = b.spectral_graph_content(d, seed)
            elif method == "kmeans_content":
                res = b.kmeans_content(d, seed)
            else:
                raise ValueError(method)
            s = b.score(d, res)
        row.update(k_used=res.k_used, modularity=s["modularity"], onmi=s["onmi"], f1=s["f1"])
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                key = (r["method"], r["dataset"], int(r["seed"]))
            except (KeyError, ValueError):
                continue
            res[key] = r
    return res


def do_run(workers):
    jobs = all_jobs()
    done = load_results()
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    total = len(jobs)
    print(f"{len(done)}/{total} already done, {len(todo)} to run, workers={workers}")
    write_progress("run", len(done), total, f"{len(todo)} jobs queued")

    new_file = not os.path.exists(RESULTS_CSV)
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

    n_done = len(done)
    try:
        if not todo:
            return
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, j): j for j in todo}
            for fut in as_completed(futs):
                row = fut.result()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                tag = row["error"] or f"onmi={row['onmi']:.4f}"
                print(f"[{n_done}/{total}] {row['method']:<24s} {row['dataset']:<10s} seed={row['seed']:<3d} "
                      f"{row['secs']:>6.1f}s  {tag}")
                if n_done % 10 == 0:
                    write_progress("run", n_done, total, f"{total - n_done} remaining")
    finally:
        f.close()
        write_progress("run", n_done, total, "run finished" if n_done >= total else "interrupted")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(x, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_tests(a, b):
    """a, b: arrays of equal length, index-matched by seed. Returns dict."""
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    t_stat, t_p = stats.ttest_rel(a, b)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    n_nonzero = int(np.sum(diff != 0))
    if n_nonzero >= 4:
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")
    return dict(
        n=n, mean_diff=mean_diff, sd_diff=sd_diff, cohens_d=cohens_d,
        t_stat=float(t_stat), t_p=float(t_p),
        wilcoxon_stat=float(w_stat) if w_stat == w_stat else float("nan"),
        wilcoxon_p=float(w_p) if w_p == w_p else float("nan"),
        n_nonzero_diffs=n_nonzero,
    )


def required_n_for_power(cohens_d, alpha=0.05, power=0.8):
    """Smallest n (paired t-test, two-sided) reaching the target power.
    For very large |d|, n=2 (the minimum for a paired test) already exceeds
    the target power and statsmodels' root-finder fails to converge (its
    search bracket doesn't go below its default floor) -- check that case
    directly rather than reporting a spurious 'inf' for a huge effect."""
    from statsmodels.stats.power import TTestPower
    if not np.isfinite(cohens_d) or abs(cohens_d) < 1e-9:
        return float("inf")
    analysis = TTestPower()
    d = abs(cohens_d)
    power_at_2 = analysis.power(effect_size=d, nobs=2, alpha=alpha, alternative="two-sided")
    if power_at_2 >= power:
        return 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            n = analysis.solve_power(effect_size=d, alpha=alpha, power=power, alternative="two-sided")
            if n is None or not np.isfinite(n):
                # Bisect manually as a fallback for the solver's non-convergence.
                lo, hi = 2.0, 5000.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if analysis.power(effect_size=d, nobs=mid, alpha=alpha, alternative="two-sided") >= power:
                        hi = mid
                    else:
                        lo = mid
                n = hi
            return float(n)
        except Exception:
            return float("nan")


def do_summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_results()
    lines = []
    lines.append("Significance test: NF-MCD default / robust vs. spectral+content / kmeans_content")
    lines.append(f"n_seeds={N_SEEDS} (seeds 0-{N_SEEDS-1}), datasets={DATASETS}")
    lines.append("")
    lines.append("Note on 'paired': NFMCD's `seed` controls FuzzyCMeans/adaptive-m-FCM init "
                  "(and, for robust, the raw-PCA solver), all via np.random.default_rng(seed) "
                  "INSIDE nf_mcd/pipeline.py's own code path. spectral_graph+content's seed feeds "
                  "sklearn's SpectralClustering(random_state=seed), and kmeans_content's feeds "
                  "sklearn's KMeans(random_state=seed) -- both INDEPENDENT RNG streams unrelated "
                  "to NFMCD's. So matching by seed index across DIFFERENT methods (e.g. robust vs "
                  "spectral+content) does not share a noise source and should NOT be expected to "
                  "reduce variance the way genuine pairing does; the paired t-test is still a valid "
                  "test on the matched differences, but its power advantage over an unpaired test is "
                  "not guaranteed here. For nfmcd_default vs nfmcd_robust specifically, both draw from "
                  "np.random.default_rng(seed) but through DIFFERENT clusterer code ('fcm' vs "
                  "'fcm_adaptive_m', different draw counts/branches), so even this comparison's "
                  "pairing is nominal (index-matched), not a shared-random-stream design.")
    lines.append("")

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5), squeeze=False)
    axes = axes[0]

    all_comparisons = []  # for the master table

    for di, dataset in enumerate(DATASETS):
        ms = methods_for(dataset)
        lines.append("=" * 100)
        lines.append(f"DATASET: {dataset}  (methods: {ms})")
        lines.append("=" * 100)

        data = {}
        for m in ms:
            vals = []
            for s in SEEDS:
                r = rows.get((m, dataset, s))
                if r is None or r.get("error"):
                    continue
                try:
                    vals.append(float(r["onmi"]))
                except (TypeError, ValueError):
                    continue
            data[m] = np.array(vals, dtype=float)

        lines.append(f"\n{'method':<26s}{'n':>4s}{'mean':>9s}{'sd':>9s}{'95% CI':>22s}")
        ax = axes[di]
        for yi, m in enumerate(ms):
            v = data[m]
            if len(v) == 0:
                lines.append(f"{m:<26s}{'0':>4s}  (no successful runs)")
                continue
            lo, hi = bootstrap_ci(v)
            lines.append(f"{m:<26s}{len(v):>4d}{v.mean():>9.4f}{v.std(ddof=1):>9.4f}"
                         f"   [{lo:.4f}, {hi:.4f}]")
            ax.errorbar(v.mean(), yi, xerr=[[v.mean() - lo], [hi - v.mean()]],
                        fmt="o", capsize=4, color=f"C{yi}")
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels(ms, fontsize=8)
        ax.set_xlabel("LFK overlapping NMI")
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=0.3)

        comparisons = [("nfmcd_robust", "nfmcd_default")]
        if "spectral_graph+content" in ms:
            comparisons.append(("spectral_graph+content", "nfmcd_robust"))
        if "kmeans_content" in ms:
            comparisons.append(("nfmcd_robust", "kmeans_content"))

        lines.append(f"\n{'comparison (A vs B)':<38s}{'n':>4s}{'mean(A-B)':>12s}{'sd(diff)':>10s}"
                     f"{'d':>8s}{'t_p':>10s}{'wilcox_p':>10s}{'n_req(d80)':>12s}")
        for a_name, b_name in comparisons:
            a, b = data.get(a_name), data.get(b_name)
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                lines.append(f"{a_name} vs {b_name}: missing data")
                continue
            n_use = min(len(a), len(b))
            res = paired_tests(a[:n_use], b[:n_use])
            n_req = required_n_for_power(res["cohens_d"])
            sig_t = "***" if res["t_p"] < 0.001 else "**" if res["t_p"] < 0.01 else "*" if res["t_p"] < 0.05 else "ns"
            n_req_str = f"{n_req:.1f}" if np.isfinite(n_req) else "inf"
            lines.append(f"{a_name+' vs '+b_name:<38s}{res['n']:>4d}{res['mean_diff']:>12.4f}"
                         f"{res['sd_diff']:>10.4f}{res['cohens_d']:>8.2f}"
                         f"{res['t_p']:>9.4f}{sig_t:>1s}{res['wilcoxon_p']:>10.4f}{n_req_str:>12s}")
            all_comparisons.append(dict(dataset=dataset, a=a_name, b=b_name, **res, n_req_power80=n_req))
        lines.append("")

    plt.tight_layout()
    plt.savefig(CI_PNG, dpi=130)
    plt.close(fig)

    lines.append("=" * 100)
    lines.append("POWER ANALYSIS: sample size that WOULD have been needed to detect the observed")
    lines.append("effect (from this n=20 run) at alpha=0.05, power=0.8, vs. this project's usual n=3")
    lines.append("=" * 100)
    lines.append(f"\n{'dataset':<12s}{'comparison':<38s}{'|d|':>8s}{'n_required':>12s}{'n=3 adequate?':>16s}")
    for c in all_comparisons:
        d_abs = abs(c["cohens_d"])
        n_req = c["n_req_power80"]
        adequate = "no" if (not np.isfinite(n_req) or n_req > 3) else "yes"
        n_req_str = f"{n_req:.1f}" if np.isfinite(n_req) else "inf"
        lines.append(f"{c['dataset']:<12s}{c['a']+' vs '+c['b']:<38s}{d_abs:>8.2f}{n_req_str:>12s}{adequate:>16s}")

    atomic_write_text(SUMMARY_LOG, "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {SUMMARY_LOG} and {CI_PNG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.cmd == "run":
        do_run(args.workers)
    elif args.cmd == "summary":
        do_summary()
