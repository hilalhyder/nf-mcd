"""
Scaling experiment for NF-MCD: wall-clock time and peak memory vs. graph size
(nf_mcd/* is not modified -- this file only reads and calls existing code).

Every single fit runs in its own subprocess (the `fit` subcommand), so a crash
or out-of-memory kill at one size can never corrupt results for other sizes or
methods, and the orchestrator (`run`) can enforce a wall-clock timeout per fit.
Results are appended to experiments/scale_results.csv one row at a time,
flushed and fsynced immediately, so the run is resumable after an interruption
-- a fresh `run` re-reads the CSV and skips anything already recorded, and
stops re-extending a method past a size where it previously failed/timed out.

Usage:
    py run_scale.py run          # Parts A + B + C, resumable
    py run_scale.py summary      # (re)build scale_summary.log + plots from the CSV
    py run_scale.py fit ...      # internal: single-fit worker, used via subprocess
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import threading

EXP = os.path.join(os.path.dirname(__file__), "experiments")
os.makedirs(EXP, exist_ok=True)
RESULTS_CSV = os.path.join(EXP, "scale_results.csv")
PROGRESS_MD = os.path.join(EXP, "scale_progress.md")
SUMMARY_LOG = os.path.join(EXP, "scale_summary.log")

FIELDS = ["part", "method", "clusterer", "n", "k", "seed", "avg_degree",
          "wall_generate_s", "wall_fit_s", "peak_rss_mb", "success", "error", "stage_json"]

FIT_TIMEOUT_S = 120
DENSE_MEM_GUARD_BYTES = 2.0e9  # proactive skip for spectral_graph_content's O(n^2) affinity

SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000]
N_COMMUNITIES = 8
TARGET_DEGREE = 18.0
RATIO = 0.02 / 0.18  # matches this project's own default p_out/p_in ratio (datasets.py)

PART_A_METHODS = ["nfmcd_default", "nfmcd_robust", "louvain", "spectral_graph_content"]
PART_C_SIZES = [1000, 5000]
PART_C_CLUSTERERS = ["fcm", "fcm_adaptive_m", "kmeans_softmax", "gmm", "spectral_soft"]


def seeds_for(n: int):
    if n <= 2000:
        return [0, 1, 2]
    if n <= 10000:
        return [0, 1]
    return [0]


def p_in_out_for(n: int, n_communities: int = N_COMMUNITIES,
                  target_degree: float = TARGET_DEGREE, ratio: float = RATIO):
    """p_in/p_out that hold expected node degree ~= target_degree as n grows,
    at a fixed p_out/p_in ratio matching the project's own SBM default."""
    block = n / n_communities
    denom = (block - 1) + ratio * (n - block)
    p_in = target_degree / max(denom, 1e-9)
    p_in = min(max(p_in, 1e-6), 1.0)
    return p_in, min(max(p_in * ratio, 1e-6), 1.0)


# ---------------------------------------------------------------------------
# worker (runs in its own subprocess)
# ---------------------------------------------------------------------------

def _make_dataset(n: int, seed: int):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    p_in, p_out = p_in_out_for(n)
    return generate_synthetic_multimodal_graph(
        n_nodes=n, n_communities=N_COMMUNITIES, p_in=p_in, p_out=p_out,
        missing_modality_rate=0.0, misalignment_rate=0.0, overlap_rate=0.0, seed=seed,
    )


def _peak_rss_tracker():
    import psutil
    proc = psutil.Process()
    state = {"peak": proc.memory_info().rss, "stop": False}

    def poll():
        while not state["stop"]:
            try:
                rss = proc.memory_info().rss
                if rss > state["peak"]:
                    state["peak"] = rss
            except Exception:
                pass
            time.sleep(0.03)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    return state, t


def _stop_tracker(state, t):
    state["stop"] = True
    t.join(timeout=2.0)
    return state["peak"]


def _fit_nfmcd_timed(data, k: int, seed: int, robust: bool, clusterer: str, stages: bool):
    from nf_mcd.pipeline import NFMCD

    kwargs = {}
    if clusterer and clusterer != "fcm":
        kwargs["clusterer"] = clusterer

    if not stages:
        t0 = time.monotonic()
        model = NFMCD.robust(n_communities=k, seed=seed, **kwargs) if robust else NFMCD(n_communities=k, seed=seed, **kwargs)
        model.fit(data.G, text_embeddings=data.text_embeddings, image_embeddings=data.image_embeddings)
        return time.monotonic() - t0, None

    # Manual replay of NFMCD.fit()'s exact stage sequence (pipeline.py, default
    # params only), timed per stage. Not used with clusterer!='fcm'/robust.
    from nf_mcd import community_detection as cd
    from nf_mcd import topology as topo
    from nf_mcd.fuzzy_fusion import NeuroFuzzyFusion, ANFISAgreement

    G, e_t, e_v = data.G, data.text_embeddings, data.image_embeddings
    common_dim, structural_dim, fcm_m = 8, k, 1.5
    alpha_min, alpha_max = 0.15, 0.85
    timings = {}

    fusion = NeuroFuzzyFusion(common_dim=common_dim, anfis=ANFISAgreement(), seed=seed)
    t0 = time.monotonic()
    fr = fusion.fuse(e_t, e_v)
    timings["fusion"] = time.monotonic() - t0

    t0 = time.monotonic()
    Z_s = topo.compute_structural_embedding(G, dim=structural_dim, seed=seed)
    timings["structural_embedding"] = time.monotonic() - t0

    t0 = time.monotonic()
    alpha = topo.compute_alpha(fr.confidence, alpha_min, alpha_max)
    fused_features = topo.fuse_features(fr.fused, Z_s, alpha)
    timings["alpha_fuse_features"] = time.monotonic() - t0
    timings["fused_feature_dim"] = fused_features.shape[1]

    t0 = time.monotonic()
    fcm = cd.FuzzyCMeans(n_clusters=k, m=fcm_m, seed=seed)
    fcm.fit(fused_features)
    timings["cluster_fcm"] = time.monotonic() - t0

    return sum(v for v in timings.values() if isinstance(v, float)), timings


def cmd_fit(args):
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        t_gen = None
        try:
            t0 = time.monotonic()
            data = _make_dataset(args.n, args.seed)
            t_gen = time.monotonic() - t0
        except Exception as exc:
            print(json.dumps({"success": False, "error": f"generate: {type(exc).__name__}: {exc}",
                               "wall_generate_s": t_gen, "wall_fit_s": None, "peak_rss_mb": None,
                               "avg_degree": None, "stage_json": None}))
            return

        avg_degree = 2.0 * data.G.number_of_edges() / max(data.G.number_of_nodes(), 1)
        state, thread = _peak_rss_tracker()
        try:
            if args.method in ("nfmcd_default", "nfmcd_robust"):
                wall_fit, stages = _fit_nfmcd_timed(
                    data, args.n_communities, args.seed,
                    robust=(args.method == "nfmcd_robust"),
                    clusterer=args.clusterer, stages=args.stages,
                )
                stage_json = json.dumps(stages) if stages is not None else None
            elif args.method in ("louvain", "spectral_graph_content"):
                from nf_mcd import baselines as bl
                d = dict(G=data.G, e_t=data.text_embeddings, e_v=data.image_embeddings,
                         n_communities=args.n_communities, true=data.true_communities)
                t0 = time.monotonic()
                (bl.louvain if args.method == "louvain" else bl.spectral_graph_content)(d, args.seed)
                wall_fit = time.monotonic() - t0
                stage_json = None
            else:
                raise ValueError(f"unknown method {args.method!r}")
            peak = _stop_tracker(state, thread)
            print(json.dumps({"success": True, "wall_generate_s": t_gen, "wall_fit_s": wall_fit,
                               "peak_rss_mb": peak / 1e6, "avg_degree": avg_degree,
                               "stage_json": stage_json, "error": ""}))
        except Exception as exc:
            peak = _stop_tracker(state, thread)
            print(json.dumps({"success": False, "wall_generate_s": t_gen, "wall_fit_s": None,
                               "peak_rss_mb": peak / 1e6, "avg_degree": avg_degree,
                               "stage_json": None, "error": f"{type(exc).__name__}: {exc}"}))


# ---------------------------------------------------------------------------
# orchestrator (this process only; launches `fit` as a subprocess)
# ---------------------------------------------------------------------------

def _read_existing():
    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return rows


def _key(part, method, clusterer, n, seed):
    return (part, method, clusterer or "", int(n), int(seed))


def _write_progress(phase: str):
    with open(PROGRESS_MD + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"# Scale experiment progress\n\n- Phase: **{phase}**\n"
                f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"## Resume (finished rows are skipped automatically)\n\n"
                f"```\ncd {os.path.dirname(__file__)}\npy run_scale.py run\npy run_scale.py summary\n```\n")
    os.replace(PROGRESS_MD + ".tmp", PROGRESS_MD)


_csv_file = None
_csv_writer = None


def _open_csv():
    global _csv_file, _csv_writer
    new = not os.path.exists(RESULTS_CSV)
    _csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=FIELDS)
    if new:
        _csv_writer.writeheader()
        _csv_file.flush()
        os.fsync(_csv_file.fileno())


def _append_row(row: dict):
    for k in FIELDS:
        row.setdefault(k, "")
    _csv_writer.writerow(row)
    _csv_file.flush()
    os.fsync(_csv_file.fileno())


def _run_fit_subprocess(part, method, n, seed, k=N_COMMUNITIES, clusterer="fcm", stages=False):
    cmd = [sys.executable, os.path.abspath(__file__), "fit",
           "--method", method, "--n", str(n), "--seed", str(seed),
           "--n-communities", str(k), "--clusterer", clusterer]
    if stages:
        cmd.append("--stages")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        row = dict(part=part, method=method, clusterer=clusterer, n=n, k=k, seed=seed,
                   success=False, error=f"timeout>{FIT_TIMEOUT_S}s (subprocess wall {wall:.1f}s)")
        return row, False
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        row = dict(part=part, method=method, clusterer=clusterer, n=n, k=k, seed=seed,
                   success=False, error=f"subprocess exit {proc.returncode}: {tail}")
        return row, False
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        row = dict(part=part, method=method, clusterer=clusterer, n=n, k=k, seed=seed,
                   success=False, error=f"bad worker output: {exc}; stdout tail: {proc.stdout[-400:]}")
        return row, False
    row = dict(part=part, method=method, clusterer=clusterer, n=n, k=k, seed=seed, **out)
    ok = bool(out.get("success")) and (out.get("wall_fit_s") or 0) <= FIT_TIMEOUT_S
    return row, ok


def cmd_run(args):
    existing = _read_existing()
    done = {_key(r["part"], r["method"], r.get("clusterer", ""), r["n"], r["seed"]) for r in existing}
    # A method that already has a failure/skip recorded at some n stops being
    # extended to any larger n, exactly like a live run would after that point.
    cutoff_n = {}  # (part, method, clusterer) -> smallest n at which it failed
    for r in existing:
        if r.get("success") != "True":
            mk = (r["part"], r["method"], r.get("clusterer", ""))
            n = int(r["n"])
            cutoff_n[mk] = min(cutoff_n.get(mk, 10 ** 12), n)

    _open_csv()

    # ---- Part A: end-to-end scaling, 4 methods across SIZES ----
    for method in PART_A_METHODS:
        for n in SIZES:
            mk = ("A", method, "")
            if n >= cutoff_n.get(mk, 10 ** 12):
                continue
            if method == "spectral_graph_content" and n * n * 8 > DENSE_MEM_GUARD_BYTES:
                key = _key("A", method, "", n, seeds_for(n)[0])
                if key not in done:
                    est_gb = n * n * 8 / 1e9
                    row = dict(part="A", method=method, clusterer="", n=n, k=N_COMMUNITIES,
                               seed=seeds_for(n)[0], success=False,
                               error=f"memory_guard: dense n*n affinity alone would need "
                                     f"~{est_gb:.1f} GB (guard={DENSE_MEM_GUARD_BYTES/1e9:.1f} GB)")
                    _append_row(row)
                    print(f"[A] {method} n={n}: skipped (memory guard, ~{est_gb:.1f} GB)")
                cutoff_n[mk] = n
                continue
            failed_here = False
            for seed in seeds_for(n):
                key = _key("A", method, "", n, seed)
                if key in done:
                    continue
                row, ok = _run_fit_subprocess("A", method, n, seed)
                _append_row(row)
                done.add(key)
                tag = "ok" if ok else "FAIL/timeout/slow"
                t = row.get("wall_fit_s")
                print(f"[A] {method} n={n} seed={seed}: {tag}"
                      f"{f' ({float(t):.2f}s)' if isinstance(t, (int, float)) else ''}")
                if not ok:
                    failed_here = True
            if failed_here:
                cutoff_n[mk] = n
            _write_progress(f"Part A: {method} n={n}")

    # ---- Part B: per-stage breakdown for nfmcd_default, at sizes that succeeded in A ----
    completed_sizes = sorted({
        int(r["n"]) for r in _read_existing()
        if r["part"] == "A" and r["method"] == "nfmcd_default" and r["success"] == "True"
    })
    for n in completed_sizes:
        key = _key("B", "nfmcd_default", "", n, 0)
        if key in done:
            continue
        row, ok = _run_fit_subprocess("B", "nfmcd_default", n, 0, stages=True)
        _append_row(row)
        done.add(key)
        print(f"[B] nfmcd_default n={n}: {'ok' if ok else 'FAIL'}")
        _write_progress(f"Part B: n={n}")

    # ---- Part C: clusterer comparison at representative sizes ----
    for n in PART_C_SIZES:
        for clusterer in PART_C_CLUSTERERS:
            for seed in [0, 1] if n <= 2000 else [0]:
                key = _key("C", "nfmcd_default", clusterer, n, seed)
                if key in done:
                    continue
                row, ok = _run_fit_subprocess("C", "nfmcd_default", n, seed, clusterer=clusterer)
                _append_row(row)
                done.add(key)
                t = row.get("wall_fit_s")
                print(f"[C] clusterer={clusterer} n={n} seed={seed}: {'ok' if ok else 'FAIL'}"
                      f"{f' ({float(t):.2f}s)' if isinstance(t, (int, float)) else ''}")
        _write_progress(f"Part C: n={n}")

    _csv_file.close()
    _write_progress("done")
    print("Run complete.")


# ---------------------------------------------------------------------------
# summary: log + plots
# ---------------------------------------------------------------------------

def cmd_summary(args):
    import numpy as np
    rows = _read_existing()
    if not rows:
        print("No results yet; run `py run_scale.py run` first.")
        return

    def f(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    lines = []
    lines.append("Scaling results (synthetic SBM graphs, n_communities=8, avg degree ~18)")
    lines.append("=" * 78)

    # --- Part A table ---
    lines.append("\nPart A: end-to-end wall-clock fit time (s) and peak RSS (MB), mean over seeds")
    lines.append("-" * 78)
    header = f"{'method':<24}{'n':>8}{'wall_fit_s':>14}{'peak_rss_mb':>14}{'avg_deg':>10}{'note':>10}"
    lines.append(header)
    slopes = {}
    for method in PART_A_METHODS:
        pts = []
        for n in SIZES:
            rs = [r for r in rows if r["part"] == "A" and r["method"] == method and int(r["n"]) == n]
            if not rs:
                continue
            oks = [r for r in rs if r["success"] == "True"]
            if oks:
                wt = np.mean([f(r, "wall_fit_s") for r in oks])
                mem = np.mean([f(r, "peak_rss_mb") for r in oks])
                deg = np.mean([f(r, "avg_degree") for r in oks if f(r, "avg_degree") is not None])
                lines.append(f"{method:<24}{n:>8}{wt:>14.3f}{mem:>14.1f}{deg:>10.1f}{'':>10}")
                pts.append((n, wt))
            else:
                err = (rs[0].get("error") or "")[:60]
                lines.append(f"{method:<24}{n:>8}{'--':>14}{'--':>14}{'--':>10}  {err}")
        if len(pts) >= 2:
            logn = np.log([p[0] for p in pts])
            logt = np.log([max(p[1], 1e-6) for p in pts])
            slope = float(np.polyfit(logn, logt, 1)[0])
            slopes[method] = slope
            lines.append(f"  -> empirical scaling exponent (log-log slope) for {method}: {slope:.2f}")

    # --- Part B table ---
    lines.append("\nPart B: nfmcd_default per-stage time share (s)")
    lines.append("-" * 78)
    stage_names = ["fusion", "structural_embedding", "alpha_fuse_features", "cluster_fcm"]
    header = f"{'n':>8}" + "".join(f"{s:>22}" for s in stage_names) + f"{'total':>10}"
    lines.append(header)
    stage_rows = sorted([r for r in rows if r["part"] == "B"], key=lambda r: int(r["n"]))
    stage_series = {s: [] for s in stage_names}
    ns_b = []
    for r in stage_rows:
        if r["success"] != "True" or not r.get("stage_json"):
            continue
        st = json.loads(r["stage_json"])
        vals = [st.get(s, float("nan")) for s in stage_names]
        total = sum(v for v in vals if isinstance(v, (int, float)))
        lines.append(f"{int(r['n']):>8}" + "".join(f"{v:>22.4f}" for v in vals) + f"{total:>10.4f}")
        ns_b.append(int(r["n"]))
        for s, v in zip(stage_names, vals):
            stage_series[s].append(v)
    if len(ns_b) >= 2:
        lines.append("  per-stage empirical scaling exponents:")
        for s in stage_names:
            vals = stage_series[s]
            pos = [(n, v) for n, v in zip(ns_b, vals) if v and v > 0]
            if len(pos) >= 2:
                logn = np.log([p[0] for p in pos])
                logt = np.log([p[1] for p in pos])
                sl = float(np.polyfit(logn, logt, 1)[0])
                lines.append(f"    {s:<24} slope={sl:.2f}")

    # --- Part C table ---
    lines.append("\nPart C: clusterer comparison (nfmcd_default with each clusterer)")
    lines.append("-" * 78)
    header = f"{'clusterer':<20}{'n':>8}{'wall_fit_s':>14}{'peak_rss_mb':>14}"
    lines.append(header)
    for n in PART_C_SIZES:
        for clusterer in PART_C_CLUSTERERS:
            rs = [r for r in rows if r["part"] == "C" and r["n"] == str(n) and r["clusterer"] == clusterer]
            oks = [r for r in rs if r["success"] == "True"]
            if oks:
                wt = np.mean([f(r, "wall_fit_s") for r in oks])
                mem = np.mean([f(r, "peak_rss_mb") for r in oks])
                lines.append(f"{clusterer:<20}{n:>8}{wt:>14.3f}{mem:>14.1f}")
            elif rs:
                lines.append(f"{clusterer:<20}{n:>8}{'--':>14}{'--':>14}  {(rs[0].get('error') or '')[:50]}")

    with open(SUMMARY_LOG + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(SUMMARY_LOG + ".tmp", SUMMARY_LOG)
    print("\n".join(lines))

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for method in PART_A_METHODS:
            pts = sorted({(int(r["n"]), f(r, "wall_fit_s")) for r in rows
                          if r["part"] == "A" and r["method"] == method and r["success"] == "True"
                          and f(r, "wall_fit_s") is not None})
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_nodes"); ax.set_ylabel("wall-clock fit time (s)")
        ax.set_title("NF-MCD scaling vs. baselines (log-log)")
        ax.axhline(FIT_TIMEOUT_S, color="gray", linestyle="--", linewidth=1, label=f"{FIT_TIMEOUT_S}s cutoff")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(EXP, "scale_time.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for method in PART_A_METHODS:
            pts = sorted({(int(r["n"]), f(r, "peak_rss_mb")) for r in rows
                          if r["part"] == "A" and r["method"] == method and r["success"] == "True"
                          and f(r, "peak_rss_mb") is not None})
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_nodes"); ax.set_ylabel("peak RSS (MB)")
        ax.set_title("Peak process memory vs. graph size (log-log)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(EXP, "scale_memory.png"), dpi=150)
        plt.close(fig)
        print("\nPlots written: scale_time.png, scale_memory.png")
    except Exception as exc:
        print(f"\n(plotting skipped: {exc})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--method", required=True)
    p_fit.add_argument("--n", type=int, required=True)
    p_fit.add_argument("--seed", type=int, required=True)
    p_fit.add_argument("--n-communities", type=int, default=N_COMMUNITIES)
    p_fit.add_argument("--clusterer", default="fcm")
    p_fit.add_argument("--stages", action="store_true")
    p_fit.set_defaults(func=cmd_fit)

    p_run = sub.add_parser("run")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summary")
    p_sum.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
