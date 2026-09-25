"""
Compound-stressor experiment: does the "robust" preset (NFMCD.robust()) still
avoid the fuzzy-c-means collapse when high community count (k=32, where the
plain default was already found to collapse in isolation -- see
experiments/qualityscale_summary.log Part B) is COMBINED with a second
stressor at the same time? nf_mcd/* is not modified; this file only reads and
calls existing code (nf_mcd.datasets, nf_mcd.baselines, run_scale.p_in_out_for,
run_missing.apply_mask; the cross-modal mismatch injection mirrors, rather
than imports, run_mechanism.py's build_B logic, since that function fetches a
real cached dataset and this experiment needs the synthetic k=32/n=8000 base
graph instead).

Base condition: n_communities=32, ~250 nodes/community (n=8000), avg degree
~18, demo.py-style content-injection defaults (missing_modality_rate=0.15,
misalignment_rate=0.15, overlap_rate=0.10) -- identical construction to
experiments/qualityscale_results.csv's Part B, k=32 rows, which are reused
directly as the "no additional stressor" anchor rather than refit.

Three compound conditions, each layered on the k=32 base:
  noise:    the SBM p_out/p_in ratio is widened 1x/2x/4x/8x beyond the
            project's own default ratio (0.02/0.18), at fixed avg degree
            (so the graph itself changes -- all 4 methods refit per level).
  missing:  run_missing.apply_mask('both', p, seed) at p in {0, 0.4, 0.8} on
            top of the (fixed) base graph.
  mismatch: a fraction q in {0, 0.2, 0.4} of paired nodes gets its image
            embedding swapped for a different-primary-community node's image
            (same scheme as run_mechanism.py's build_B), on top of the
            (fixed) base graph.

Each fit runs in its own subprocess (crash/OOM isolation, matching
run_scale.py/run_qualityscale.py). Results are appended to
experiments/compound_results.csv one row at a time, flushed+fsynced, so the
run is resumable.

Usage:
    py run_compound.py run          # all three conditions, resumable
    py run_compound.py summary      # (re)build summary.log + plots
    py run_compound.py fit ...      # internal: single-fit worker
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
RESULTS_CSV = os.path.join(EXP, "compound_results.csv")
PROGRESS_MD = os.path.join(EXP, "compound_progress.md")
SUMMARY_LOG = os.path.join(EXP, "compound_summary.log")
QUALITYSCALE_CSV = os.path.join(EXP, "qualityscale_results.csv")

FIELDS = ["condition", "severity", "method", "seed", "k", "n", "avg_degree",
          "onmi", "modularity", "f1", "mean_umax", "collapsed",
          "wall_fit_s", "peak_rss_mb", "success", "error"]

FIT_TIMEOUT_S = 180
DENSE_MEM_GUARD_BYTES = 2.0e9  # same guard used in run_scale.py / run_qualityscale.py

BASE_N = 8000
BASE_K = 32
BASE_RATIO = 0.02 / 0.18  # this project's own default SBM p_out/p_in ratio
TARGET_DEGREE = 18.0
UNIFORM_MARGIN = 0.05     # mean_umax within [1/k, 1/k + margin] => flagged "collapsed"

SEEDS = [0, 1, 2]
NOISE_RATIO_MULT = [1.0, 2.0, 4.0, 8.0]
MISSING_P = [0.0, 0.4, 0.8]
MISMATCH_Q = [0.0, 0.2, 0.4]

METHODS = ["nfmcd_default", "nfmcd_robust", "louvain", "spectral_graph_content"]


# ---------------------------------------------------------------------------
# dataset construction (worker side)
# ---------------------------------------------------------------------------

def _base_dataset_dict(seed: int, ratio_mult: float = 1.0):
    from nf_mcd.datasets import generate_synthetic_multimodal_graph
    from run_scale import p_in_out_for
    ratio = min(BASE_RATIO * ratio_mult, 0.999)
    p_in, p_out = p_in_out_for(BASE_N, n_communities=BASE_K, target_degree=TARGET_DEGREE, ratio=ratio)
    data = generate_synthetic_multimodal_graph(
        n_nodes=BASE_N, n_communities=BASE_K, p_in=p_in, p_out=p_out, seed=seed,
    )
    return dict(G=data.G, e_t=data.text_embeddings, e_v=data.image_embeddings,
                true=data.true_communities, n_communities=BASE_K)


def _inject_mismatch(d, q: float, seed: int):
    """Mirrors run_mechanism.py's build_B: swap a fraction q of paired nodes'
    image embedding for a different-primary-community node's image."""
    import numpy as np
    n = d["G"].number_of_nodes()
    primary = np.array([min(t) for t in d["true"]])
    e_t, e_v = d["e_t"], d["e_v"]
    paired = np.array([e_t[i] is not None and e_v[i] is not None for i in range(n)])
    has_img = np.array([e_v[i] is not None for i in range(n)])
    rng = np.random.default_rng(4000 + seed)
    u = rng.random(n)
    img_idx = np.where(has_img)[0]
    donors = np.full(n, -1)
    for i in range(n):
        cand = img_idx[primary[img_idx] != primary[i]]
        if len(cand):
            donors[i] = rng.choice(cand)
    bad = paired & (u < q) & (donors >= 0)
    e_v2 = list(e_v)
    for i in np.where(bad)[0]:
        e_v2[i] = e_v[donors[i]]
    d2 = dict(d)
    d2["e_v"] = e_v2
    return d2


def _build_dataset(condition: str, severity: float, seed: int):
    if condition == "noise":
        return _base_dataset_dict(seed, ratio_mult=severity)
    d = _base_dataset_dict(seed, ratio_mult=1.0)
    if condition == "missing":
        from run_missing import apply_mask
        return apply_mask(d, "both", severity, seed)
    if condition == "mismatch":
        return _inject_mismatch(d, severity, seed)
    raise ValueError(f"unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# worker (runs in its own subprocess)
# ---------------------------------------------------------------------------

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
            d = _build_dataset(args.condition, args.severity, args.seed)
        except Exception as exc:
            print(json.dumps({"success": False, "error": f"generate: {type(exc).__name__}: {exc}"}))
            return

        avg_degree = 2.0 * d["G"].number_of_edges() / max(d["G"].number_of_nodes(), 1)
        if args.method == "spectral_graph_content":
            n = d["G"].number_of_nodes()
            est = n * n * 8
            if est > DENSE_MEM_GUARD_BYTES:
                print(json.dumps({"success": False, "avg_degree": avg_degree,
                                   "error": f"memory_guard: ~{est/1e9:.2f} GB estimated"}))
                return

        state, thread = _peak_rss_tracker()
        try:
            from nf_mcd import baselines as bl
            t0 = time.monotonic()
            mean_umax = None
            if args.method == "nfmcd_default":
                from nf_mcd.pipeline import NFMCD
                model = NFMCD(n_communities=BASE_K, seed=args.seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = bl._nfmcd_result(model)
                mean_umax = float(model.U_.max(axis=1).mean())
            elif args.method == "nfmcd_robust":
                from nf_mcd.pipeline import NFMCD
                model = NFMCD.robust(n_communities=BASE_K, seed=args.seed)
                model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
                res = bl._nfmcd_result(model)
                mean_umax = float(model.U_.max(axis=1).mean())
            elif args.method == "louvain":
                res = bl.louvain(d, args.seed)
            elif args.method == "spectral_graph_content":
                res = bl.spectral_graph_content(d, args.seed)
            else:
                raise ValueError(f"unknown method {args.method!r}")
            wall_fit = time.monotonic() - t0
            s = bl.score(d, res)
            peak = _stop_tracker(state, thread)
            collapsed = None
            if mean_umax is not None:
                collapsed = bool(mean_umax < (1.0 / BASE_K) + UNIFORM_MARGIN)
            print(json.dumps({"success": True, "wall_fit_s": wall_fit, "peak_rss_mb": peak / 1e6,
                               "avg_degree": avg_degree, "onmi": s["onmi"], "modularity": s["modularity"],
                               "f1": s["f1"], "mean_umax": mean_umax, "collapsed": collapsed, "error": ""}))
        except Exception as exc:
            peak = _stop_tracker(state, thread)
            print(json.dumps({"success": False, "wall_fit_s": None, "peak_rss_mb": peak / 1e6,
                               "avg_degree": avg_degree, "error": f"{type(exc).__name__}: {exc}"}))


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def _read_existing():
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def _key(condition, severity, method, seed):
    return (condition, float(severity), method, int(seed))


def _write_progress(phase: str):
    with open(PROGRESS_MD + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"# Compound-stressor experiment progress\n\n- Phase: **{phase}**\n"
                f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"## Resume (finished rows are skipped automatically)\n\n"
                f"```\ncd {os.path.dirname(__file__)}\npy run_compound.py run\n"
                f"py run_compound.py summary\n```\n")
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


def _run_fit_subprocess(condition, severity, method, seed):
    cmd = [sys.executable, os.path.abspath(__file__), "fit",
           "--condition", condition, "--severity", str(severity),
           "--method", method, "--seed", str(seed)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        return dict(condition=condition, severity=severity, method=method, seed=seed, k=BASE_K, n=BASE_N,
                    success=False, error=f"timeout>{FIT_TIMEOUT_S}s (subprocess wall {wall:.1f}s)"), False
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        return dict(condition=condition, severity=severity, method=method, seed=seed, k=BASE_K, n=BASE_N,
                    success=False, error=f"subprocess exit {proc.returncode}: {tail}"), False
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return dict(condition=condition, severity=severity, method=method, seed=seed, k=BASE_K, n=BASE_N,
                    success=False, error=f"bad worker output: {exc}; stdout tail: {proc.stdout[-400:]}"), False
    row = dict(condition=condition, severity=severity, method=method, seed=seed, k=BASE_K, n=BASE_N, **out)
    ok = bool(out.get("success"))
    return row, ok


CONDITIONS = [
    ("noise", NOISE_RATIO_MULT),
    ("missing", MISSING_P),
    ("mismatch", MISMATCH_Q),
]


def cmd_run(args):
    existing = _read_existing()
    done = {_key(r["condition"], r["severity"], r["method"], r["seed"]) for r in existing}
    _open_csv()

    for condition, severities in CONDITIONS:
        for method in METHODS:
            for severity in severities:
                for seed in SEEDS:
                    key = _key(condition, severity, method, seed)
                    if key in done:
                        continue
                    row, ok = _run_fit_subprocess(condition, severity, method, seed)
                    _append_row(row)
                    done.add(key)
                    tag = "ok" if ok else "FAIL/timeout"
                    onmi = row.get("onmi")
                    umax = row.get("mean_umax")
                    extra = f" onmi={float(onmi):.3f}" if isinstance(onmi, (int, float)) else ""
                    extra += f" umax={float(umax):.3f}" if isinstance(umax, (int, float)) and umax not in (None, "") else ""
                    print(f"[{condition}] {method} severity={severity} seed={seed}: {tag}{extra}")
                _write_progress(f"{condition}: {method} severity={severity}")

    _csv_file.close()
    _write_progress("done")
    print("Run complete.")


# ---------------------------------------------------------------------------
# summary: log + plots
# ---------------------------------------------------------------------------

def _qualityscale_k32_anchor():
    """Pull k=32 (n=8000) rows from qualityscale_results.csv Part B as the
    'no additional stressor' reference -- identical construction to this
    experiment's base graph (same p_in_out_for call, same content defaults)."""
    if not os.path.exists(QUALITYSCALE_CSV):
        return {}
    with open(QUALITYSCALE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r.get("part") == "B" and r.get("k") == "32" and r.get("success") == "True":
            out.setdefault(r["method"], []).append(r)
    return out


def cmd_summary(args):
    import numpy as np
    rows = _read_existing()
    if not rows:
        print("No results yet; run `py run_compound.py run` first.")
        return

    def f(r, key):
        v = r.get(key, "")
        if v in ("", None):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    anchor = _qualityscale_k32_anchor()

    lines = []
    lines.append("Compound-stressor experiment: k=32 (n=8000, ~250 nodes/community) plus a")
    lines.append("second stressor layered on top. 'collapsed' flags mean_umax within")
    lines.append(f"[1/32={1/32:.4f}, {1/32 + UNIFORM_MARGIN:.4f}] (uniform-membership degeneracy).")
    lines.append("=" * 92)

    if anchor:
        lines.append("\nAnchor: k=32 alone, no additional stressor (pulled from qualityscale_results.csv Part B)")
        lines.append("-" * 92)
        for method in METHODS:
            rs = anchor.get(method, [])
            if not rs:
                continue
            onmi = np.mean([float(r["onmi"]) for r in rs])
            mod = np.mean([float(r["modularity"]) for r in rs])
            lines.append(f"  {method:<24} onmi={onmi:.4f}  modularity={mod:.4f}  (n={len(rs)} seeds)")

    def table(condition, severities, sev_label):
        lines.append(f"\nCondition: {condition} (severity = {sev_label})")
        lines.append("-" * 92)
        header = f"{'method':<24}{sev_label:>12}{'onmi':>10}{'modularity':>12}{'f1':>10}{'mean_umax':>12}{'collapsed':>12}"
        lines.append(header)
        for method in METHODS:
            pts = []
            for sev in severities:
                rs = [r for r in rows if r["condition"] == condition and r["method"] == method
                      and f(r, "severity") == sev]
                oks = [r for r in rs if r["success"] == "True"]
                if oks:
                    onmi = np.mean([f(r, "onmi") for r in oks])
                    mod = np.mean([f(r, "modularity") for r in oks])
                    f1 = np.mean([f(r, "f1") for r in oks])
                    umax_vals = [f(r, "mean_umax") for r in oks if f(r, "mean_umax") is not None]
                    umax = np.mean(umax_vals) if umax_vals else None
                    coll_vals = [r.get("collapsed") for r in oks if r.get("collapsed") not in ("", None)]
                    coll = "yes" if coll_vals and all(v == "True" for v in coll_vals) else (
                        "partial" if coll_vals and any(v == "True" for v in coll_vals) else
                        ("no" if coll_vals else "--"))
                    umax_s = f"{umax:.4f}" if umax is not None else "--"
                    lines.append(f"{method:<24}{sev:>12}{onmi:>10.4f}{mod:>12.4f}{f1:>10.4f}{umax_s:>12}{coll:>12}")
                    pts.append((sev, onmi))
                else:
                    err = ((rs[0].get("error") if rs else "") or "")[:40]
                    lines.append(f"{method:<24}{sev:>12}{'--':>10}{'--':>12}{'--':>10}{'--':>12}{'--':>12}  {err}")
            if len(pts) >= 2:
                delta = pts[-1][1] - pts[0][1]
                lines.append(f"  -> {method}: ONMI at severity={pts[0][0]}: {pts[0][1]:.4f}, "
                              f"at severity={pts[-1][0]}: {pts[-1][1]:.4f}, delta={delta:+.4f}")
        # robust-vs-default gap at each severity
        gaps = []
        for sev in severities:
            d_rs = [r for r in rows if r["condition"] == condition and r["method"] == "nfmcd_default"
                    and f(r, "severity") == sev and r["success"] == "True"]
            r_rs = [r for r in rows if r["condition"] == condition and r["method"] == "nfmcd_robust"
                    and f(r, "severity") == sev and r["success"] == "True"]
            if d_rs and r_rs:
                gap = np.mean([f(r, "onmi") for r in r_rs]) - np.mean([f(r, "onmi") for r in d_rs])
                gaps.append((sev, gap))
        if gaps:
            lines.append("  robust-minus-default ONMI gap by severity: " +
                          ", ".join(f"{s}: {g:+.4f}" for s, g in gaps))

    table("noise", NOISE_RATIO_MULT, "ratio_mult")
    table("missing", MISSING_P, "p")
    table("mismatch", MISMATCH_Q, "q")

    with open(SUMMARY_LOG + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(SUMMARY_LOG + ".tmp", SUMMARY_LOG)
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for condition, severities, fname, xlabel, title in (
            ("noise", NOISE_RATIO_MULT, "compound_noise.png", "p_out/p_in ratio multiplier",
             "k=32, n=8000: quality vs. added structural noise"),
            ("missing", MISSING_P, "compound_missing.png", "fraction of nodes missing both modalities (p)",
             "k=32, n=8000: quality vs. added missing content"),
            ("mismatch", MISMATCH_Q, "compound_mismatch.png", "fraction of paired nodes with swapped image (q)",
             "k=32, n=8000: quality vs. added cross-modal mismatch"),
        ):
            fig, ax = plt.subplots(figsize=(7, 5))
            for method in METHODS:
                pts = []
                for sev in severities:
                    rs = [r for r in rows if r["condition"] == condition and r["method"] == method
                          and f(r, "severity") == sev and r["success"] == "True"]
                    if rs:
                        pts.append((sev, np.mean([f(r, "onmi") for r in rs])))
                pts.sort()
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=method)
            ax.set_xlabel(xlabel); ax.set_ylabel("LFK overlapping NMI")
            ax.set_title(title)
            ax.set_ylim(-0.02, 1.0)
            ax.axhline(1.0 / BASE_K, color="gray", linestyle=":", linewidth=1, label=f"1/k = {1/BASE_K:.3f}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(EXP, fname), dpi=150)
            plt.close(fig)
        print("\nPlots written: compound_noise.png, compound_missing.png, compound_mismatch.png")
    except Exception as exc:
        print(f"\n(plotting skipped: {exc})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--condition", required=True)
    p_fit.add_argument("--severity", type=float, required=True)
    p_fit.add_argument("--method", required=True)
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
