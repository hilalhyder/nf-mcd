"""
Clustering QUALITY vs. graph size for NF-MCD (nf_mcd/* is not modified --
this file only reads and calls existing code, plus run_scale.py for the
degree-holding p_in/p_out calibration and the subprocess-per-fit pattern).

The earlier scaling experiment (experiments/scale_summary.log) measured only
wall-clock time and peak memory as n_nodes grew to 20,000. It never checked
whether ACCURACY holds up at that scale -- every accuracy result elsewhere in
this project was measured on graphs of 146-2000 real nodes or 120-600
synthetic nodes. This experiment uses the synthetic generator's known ground
truth (no large real dataset is available here) to answer that directly, with
demo.py's realistic content-injection defaults (missing/misaligned/overlap
content), not an idealized clean case.

Part A: n_communities fixed at 8, n_nodes grows 300 -> 20,000.
Part B: per-community size fixed at ~250, n_communities grows 4 -> 50 (so
        n_nodes grows 1,000 -> 12,500) -- isolates k-count effects from n.
Part C: read off Part A/B's data for spectral_graph+content (no new fits).

Each fit runs in its own subprocess so a crash/OOM at one size can't corrupt
other results. Results are appended to experiments/qualityscale_results.csv
one row at a time, flushed+fsynced, so the run is resumable.

Usage:
    py run_qualityscale.py run          # Parts A + B, resumable
    py run_qualityscale.py summary      # (re)build summary.log + plots
    py run_qualityscale.py fit ...      # internal: single-fit worker
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
RESULTS_CSV = os.path.join(EXP, "qualityscale_results.csv")
PROGRESS_MD = os.path.join(EXP, "qualityscale_progress.md")
SUMMARY_LOG = os.path.join(EXP, "qualityscale_summary.log")

FIELDS = ["part", "method", "n", "k", "seed", "avg_degree",
          "onmi", "modularity", "f1", "n_pred",
          "wall_fit_s", "peak_rss_mb", "success", "error"]

FIT_TIMEOUT_S = 180
DENSE_MEM_GUARD_BYTES = 2.0e9  # same guard run_scale.py used for spectral_graph_content's O(n^2) affinity

PART_A_SIZES = [300, 1000, 3000, 10000, 20000]
PART_A_K = 8

PART_B_K_VALUES = [4, 8, 16, 32, 50]
PART_B_PER_COMM = 250

METHODS = ["nfmcd_default", "nfmcd_robust", "louvain", "spectral_graph_content"]


def seeds_for(n: int):
    if n <= 2000:
        return [0, 1, 2]
    if n <= 10000:
        return [0, 1]
    return [0]


# ---------------------------------------------------------------------------
# worker (runs in its own subprocess)
# ---------------------------------------------------------------------------

def _make_dataset(n: int, k: int, seed: int):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    from run_scale import p_in_out_for
    p_in, p_out = p_in_out_for(n, n_communities=k)
    # demo.py's realistic defaults (missing/misaligned/overlapping content) --
    # deliberately NOT overridden to 0, unlike run_scale.py's clean-case runs.
    return generate_synthetic_multimodal_graph(
        n_nodes=n, n_communities=k, p_in=p_in, p_out=p_out, seed=seed,
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


def cmd_fit(args):
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            data = _make_dataset(args.n, args.k, args.seed)
        except Exception as exc:
            print(json.dumps({"success": False, "error": f"generate: {type(exc).__name__}: {exc}"}))
            return

        d = dict(G=data.G, e_t=data.text_embeddings, e_v=data.image_embeddings,
                  true=data.true_communities, n_communities=args.k)
        avg_degree = 2.0 * d["G"].number_of_edges() / max(d["G"].number_of_nodes(), 1)

        state, thread = _peak_rss_tracker()
        try:
            from nf_mcd import baselines as bl
            t0 = time.monotonic()
            if args.method == "nfmcd_default":
                res = bl.nfmcd_full(d, args.seed)
            elif args.method == "nfmcd_robust":
                from nf_mcd.pipeline import NFMCD
                model = NFMCD.robust(n_communities=args.k, seed=args.seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = bl._nfmcd_result(model)
            elif args.method == "louvain":
                res = bl.louvain(d, args.seed)
            elif args.method == "spectral_graph_content":
                res = bl.spectral_graph_content(d, args.seed)
            else:
                raise ValueError(f"unknown method {args.method!r}")
            wall_fit = time.monotonic() - t0
            s = bl.score(d, res)
            peak = _stop_tracker(state, thread)
            print(json.dumps({"success": True, "wall_fit_s": wall_fit, "peak_rss_mb": peak / 1e6,
                               "avg_degree": avg_degree, "onmi": s["onmi"], "modularity": s["modularity"],
                               "f1": s["f1"], "n_pred": s["n_pred"], "error": ""}))
        except Exception as exc:
            peak = _stop_tracker(state, thread)
            print(json.dumps({"success": False, "wall_fit_s": None, "peak_rss_mb": peak / 1e6,
                               "avg_degree": avg_degree, "error": f"{type(exc).__name__}: {exc}"}))


# ---------------------------------------------------------------------------
# orchestrator (this process only; launches `fit` as a subprocess)
# ---------------------------------------------------------------------------

def _read_existing():
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def _key(part, method, n, seed):
    return (part, method, int(n), int(seed))


def _write_progress(phase: str):
    with open(PROGRESS_MD + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"# Quality-vs-scale experiment progress\n\n- Phase: **{phase}**\n"
                f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"## Resume (finished rows are skipped automatically)\n\n"
                f"```\ncd {os.path.dirname(__file__)}\npy run_qualityscale.py run\n"
                f"py run_qualityscale.py summary\n```\n")
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


def _run_fit_subprocess(part, method, n, k, seed):
    cmd = [sys.executable, os.path.abspath(__file__), "fit",
           "--method", method, "--n", str(n), "--k", str(k), "--seed", str(seed)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        return dict(part=part, method=method, n=n, k=k, seed=seed, success=False,
                    error=f"timeout>{FIT_TIMEOUT_S}s (subprocess wall {wall:.1f}s)"), False
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        return dict(part=part, method=method, n=n, k=k, seed=seed, success=False,
                    error=f"subprocess exit {proc.returncode}: {tail}"), False
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return dict(part=part, method=method, n=n, k=k, seed=seed, success=False,
                    error=f"bad worker output: {exc}; stdout tail: {proc.stdout[-400:]}"), False
    row = dict(part=part, method=method, n=n, k=k, seed=seed, **out)
    ok = bool(out.get("success"))
    return row, ok


def _run_grid(part, sizes_and_k, existing_rows, done, cutoff_n):
    """sizes_and_k: list of (n, k) pairs for this part."""
    for method in METHODS:
        for n, k in sizes_and_k:
            mk = (part, method)
            if n >= cutoff_n.get(mk, 10 ** 12):
                continue
            if method == "spectral_graph_content" and n * n * 8 > DENSE_MEM_GUARD_BYTES:
                key = _key(part, method, n, seeds_for(n)[0])
                if key not in done:
                    est_gb = n * n * 8 / 1e9
                    row = dict(part=part, method=method, n=n, k=k, seed=seeds_for(n)[0], success=False,
                               error=f"memory_guard: dense n*n affinity alone would need "
                                     f"~{est_gb:.1f} GB (guard={DENSE_MEM_GUARD_BYTES/1e9:.1f} GB)")
                    _append_row(row)
                    print(f"[{part}] {method} n={n} k={k}: skipped (memory guard, ~{est_gb:.1f} GB)")
                cutoff_n[mk] = n
                continue
            failed_here = False
            for seed in seeds_for(n):
                key = _key(part, method, n, seed)
                if key in done:
                    continue
                row, ok = _run_fit_subprocess(part, method, n, k, seed)
                _append_row(row)
                done.add(key)
                tag = "ok" if ok else "FAIL/timeout"
                t = row.get("wall_fit_s")
                onmi = row.get("onmi")
                extra = f" onmi={float(onmi):.3f}" if isinstance(onmi, (int, float)) else ""
                print(f"[{part}] {method} n={n} k={k} seed={seed}: {tag}"
                      f"{f' ({float(t):.2f}s)' if isinstance(t, (int, float)) else ''}{extra}")
                if not ok:
                    failed_here = True
            if failed_here:
                cutoff_n[mk] = n
            _write_progress(f"Part {part}: {method} n={n}")


def cmd_run(args):
    existing = _read_existing()
    done = {_key(r["part"], r["method"], r["n"], r["seed"]) for r in existing}
    cutoff_n = {}
    for r in existing:
        if r.get("success") != "True":
            mk = (r["part"], r["method"])
            n = int(r["n"])
            cutoff_n[mk] = min(cutoff_n.get(mk, 10 ** 12), n)

    _open_csv()

    part_a = [(n, PART_A_K) for n in PART_A_SIZES]
    _run_grid("A", part_a, existing, done, cutoff_n)

    part_b = [(k * PART_B_PER_COMM, k) for k in PART_B_K_VALUES]
    _run_grid("B", part_b, existing, done, cutoff_n)

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
        print("No results yet; run `py run_qualityscale.py run` first.")
        return

    def f(r, key):
        v = r.get(key, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    lines = []
    lines.append("Clustering quality vs. graph size (synthetic, demo.py-realistic content injection)")
    lines.append("=" * 88)

    def table(part, x_values, x_label):
        lines.append(f"\nPart {part}: LFK ONMI / modularity / F1 vs {x_label}")
        lines.append("-" * 88)
        header = f"{'method':<24}{x_label:>8}{'onmi':>10}{'modularity':>12}{'f1':>10}{'avg_deg':>10}"
        lines.append(header)
        for method in METHODS:
            pts = []
            for x in x_values:
                rs = [r for r in rows if r["part"] == part and r["method"] == method and int(r["n"]) == x]
                if not rs:
                    continue
                oks = [r for r in rs if r["success"] == "True"]
                if oks:
                    onmi = np.mean([f(r, "onmi") for r in oks])
                    mod = np.mean([f(r, "modularity") for r in oks])
                    f1 = np.mean([f(r, "f1") for r in oks])
                    deg = np.mean([f(r, "avg_degree") for r in oks if f(r, "avg_degree") is not None])
                    lines.append(f"{method:<24}{x:>8}{onmi:>10.4f}{mod:>12.4f}{f1:>10.4f}{deg:>10.1f}")
                    pts.append((x, onmi))
                else:
                    err = (rs[0].get("error") or "")[:50]
                    lines.append(f"{method:<24}{x:>8}{'--':>10}{'--':>12}{'--':>10}  {err}")
            if len(pts) >= 2:
                delta = pts[-1][1] - pts[0][1]
                lines.append(f"  -> {method}: ONMI at smallest={pts[0][1]:.4f}, at largest={pts[-1][1]:.4f}, "
                              f"delta={delta:+.4f}")
        return

    table("A", PART_A_SIZES, "n_nodes")
    b_x = [k * PART_B_PER_COMM for k in PART_B_K_VALUES]
    table("B", b_x, "n_nodes")

    # --- Part C: read spectral_graph_content's largest completed sizes off A/B ---
    lines.append("\nPart C: spectral_graph+content quality at its largest COMPLETED size (read off A/B, no new fits)")
    lines.append("-" * 88)
    for part, x_values in (("A", PART_A_SIZES), ("B", b_x)):
        oks = [r for r in rows if r["part"] == part and r["method"] == "spectral_graph_content"
               and r["success"] == "True"]
        if not oks:
            lines.append(f"  Part {part}: no completed spectral_graph_content rows")
            continue
        by_n = sorted({int(r["n"]) for r in oks})
        largest = by_n[-1]
        largest_rows = [r for r in oks if int(r["n"]) == largest]
        onmi_at_largest = np.mean([f(r, "onmi") for r in largest_rows])
        onmi_at_smallest = np.mean([f(r, "onmi") for r in oks if int(r["n"]) == by_n[0]])
        failed = [r for r in rows if r["part"] == part and r["method"] == "spectral_graph_content"
                  and r["success"] != "True"]
        next_size = min((int(r["n"]) for r in failed), default=None)
        lines.append(f"  Part {part}: completes up to n={largest} (ONMI={onmi_at_largest:.4f}, "
                      f"vs ONMI={onmi_at_smallest:.4f} at n={by_n[0]}); "
                      f"{'first failure at n=' + str(next_size) if next_size else 'no failure recorded'}")

    with open(SUMMARY_LOG + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(SUMMARY_LOG + ".tmp", SUMMARY_LOG)
    print("\n".join(lines))

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for part, x_values, fname, title in (
            ("A", PART_A_SIZES, "qualityscale_partA.png", "Quality vs n_nodes (k=8 fixed)"),
            ("B", b_x, "qualityscale_partB.png", "Quality vs n_nodes (~250 nodes/community, k grows)"),
        ):
            fig, ax = plt.subplots(figsize=(7, 5))
            for method in METHODS:
                pts = sorted({(int(r["n"]), f(r, "onmi")) for r in rows
                              if r["part"] == part and r["method"] == method and r["success"] == "True"
                              and f(r, "onmi") is not None})
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=method)
            ax.set_xscale("log")
            ax.set_xlabel("n_nodes"); ax.set_ylabel("LFK overlapping NMI")
            ax.set_title(title)
            ax.set_ylim(-0.02, 1.0)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(EXP, fname), dpi=150)
            plt.close(fig)
        print("\nPlots written: qualityscale_partA.png, qualityscale_partB.png")
    except Exception as exc:
        print(f"\n(plotting skipped: {exc})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--method", required=True)
    p_fit.add_argument("--n", type=int, required=True)
    p_fit.add_argument("--k", type=int, required=True)
    p_fit.add_argument("--seed", type=int, required=True)
    p_fit.set_defaults(func=cmd_fit)

    p_run = sub.add_parser("run")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summary")
    p_sum.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
