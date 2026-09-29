"""
Experiment 1 (reviewer request): controlled synthetic factor-sweep /
"performance surface" over cross-modal agreement x missingness, at two
representative structure/content regimes, comparing six methods:
structure-only, content-only, simple concatenation (no fusion mechanism),
spectral+content, fixed-alpha NF-MCD, adaptive-alpha NF-MCD.

Why a local dataset generator instead of nf_mcd.datasets directly
--------------------------------------------------------------------------
nf_mcd.datasets.generate_synthetic_multimodal_graph already exposes
independently controllable missingness (missing_modality_rate) and
cross-modal agreement (via misalignment_rate: the two are related by
agreement = 1 - misalignment_rate, since a "misaligned" node's image is
drawn from a different community's centroid, i.e. its cross-modal
similarity to the text is no better than chance). It does NOT let text and
image "signal strength" (centroid_scale) vary independently -- both
modalities share one `centroid_scale`. Per this task's constraints (another
agent is working in nf_mcd/ and other run_*.py files in the same,
non-git-worktree-isolated directory; only new files may be touched), this
module does not edit nf_mcd/datasets.py. Instead `generate_factor_graph`
below is a local, self-contained re-derivation of the same generator logic
(identical SBM graph construction, identical per-node draw order/shape),
parameterised with SEPARATE text_centroid_scale / image_centroid_scale so
"text signal" and "image signal" really are independent knobs, as the
reviewer asked. It is intentionally not wired back into nf_mcd/datasets.py.

Design
------
Two representative regimes (structural signal x content signal), each
sweeping the two primary "surface" axes -- cross-modal agreement (5 levels)
and missingness (5 levels) -- at 5 seeds:
  - strong_struct_weak_content: well-separated SBM blocks, weak (near-noise)
    text/image centroid separation.
  - weak_struct_strong_content: barely-separated SBM blocks (p_in close to
    p_out), strongly separated text/image centroids.
`calibrate` (below) verifies these two regimes actually produce the intended
asymmetry (structure-only >> content-only in one, reversed in the other)
before the full sweep is trusted.

Six methods per cell (k = ground truth, hard partition unless noted):
  structure_only   - nf_mcd.baselines.nfmcd_structure_only (NF-MCD's own
                      structure-only ablation: content masked to None both
                      modalities; equivalent to fuzzy c-means on the
                      spectral structural embedding alone, since alpha is
                      then pinned to alpha_min for every node)
  content_only     - nf_mcd.baselines.fcm_content (fuzzy c-means directly on
                      concatenated raw text+image, no structure, no fusion)
  simple_concat    - local: structural embedding and raw content
                      concatenated with NO CCA/ANFIS fusion and no alpha
                      weighting, then fuzzy c-means (the "naive baseline"
                      the reviewer asked to distinguish from spectral+content)
  spectral_content - nf_mcd.baselines.spectral_graph_content (adjacency +
                      kNN content graph, spectral clustering) -- reused
                      verbatim, this project's strongest baseline throughout
  nfmcd_alpha_fixed - nf_mcd.baselines.nfmcd_alpha_fixed (alpha_min=alpha_max=0.5)
  nfmcd_adaptive   - nf_mcd.baselines.nfmcd_full (NF-MCD defaults, adaptive
                      per-node alpha from fusion confidence)

Metrics: LFK overlapping NMI (mx.overlapping_nmi) and modularity, via
nf_mcd.baselines.score (identical scoring used by every other experiment in
this repo).

Crash-safe: every (regime, agreement, missingness, seed, method) row is
appended to experiments/factor_sweep_results.csv with flush+fsync; reruns
skip finished rows. Other files (summary log, progress, plots) are written
temp+rename.

Usage (from nfmcd_impl/):
    py run_factor_sweep.py calibrate            # sanity-check the 2 regimes
    py run_factor_sweep.py run [--workers N]     # resumable
    py run_factor_sweep.py summary               # rebuild summary log + plots
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "experiments")
RESULTS_CSV = os.path.join(EXP, "factor_sweep_results.csv")
SUMMARY_LOG = os.path.join(EXP, "factor_sweep_summary.log")
PROGRESS_MD = os.path.join(EXP, "factor_sweep_progress.md")
CALIBRATE_LOG = os.path.join(EXP, "factor_sweep_calibration.log")

N_NODES = 320
N_COMM = 4

REGIMES = {
    "strong_struct_weak_content": dict(
        p_in=0.22, p_out=0.02, text_scale=0.35, image_scale=0.35, noise_scale=1.0,
    ),
    "weak_struct_strong_content": dict(
        p_in=0.055, p_out=0.045, text_scale=4.0, image_scale=4.0, noise_scale=1.0,
    ),
}

AGREEMENT_LEVELS = (0.9, 0.7, 0.5, 0.3, 0.1)   # -> misalignment_rate = 1 - agreement
MISSING_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
SEEDS = (0, 1, 2, 3, 4)
METHODS = ["structure_only", "content_only", "simple_concat", "spectral_content",
           "nfmcd_alpha_fixed", "nfmcd_adaptive"]

FIELDS = ["regime", "agreement", "missingness", "seed", "method",
          "onmi", "modularity", "f1", "k_used", "error", "secs"]


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Factor-sweep (performance surface) progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_factor_sweep.py run --workers 4\n"
        "py run_factor_sweep.py summary\n"
        "```\n\n"
        "Results: experiments/factor_sweep_results.csv (append-only, fsynced per job). "
        "Summary: experiments/factor_sweep_summary.log.\n"
    ))


def rkey(r):
    return (r["regime"], f"{float(r['agreement']):.2f}", f"{float(r['missingness']):.2f}",
            int(r["seed"]), r["method"])


def load_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != FIELDS:
            return []
        return [dict(zip(FIELDS, r)) for r in rd if len(r) == len(FIELDS)]


def open_append():
    new = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not new:
        with open(RESULTS_CSV, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(RESULTS_CSV, "r+b") as f:
                f.truncate(cut)
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return f, w


# ---------------------------------------------------------------------------
# local dataset generator (see module docstring for why this isn't in
# nf_mcd/datasets.py)
# ---------------------------------------------------------------------------

def generate_factor_graph(
    n_nodes: int, n_communities: int,
    p_in: float, p_out: float,
    text_centroid_scale: float, image_centroid_scale: float, noise_scale: float,
    missing_modality_rate: float, misalignment_rate: float,
    text_dim: int = 384, image_dim: int = 512,
    seed: int = 0,
):
    """Same construction as nf_mcd.datasets.generate_synthetic_multimodal_graph
    (SBM graph; per-node text/image = community centroid + Gaussian noise;
    misalignment_rate fraction get an image centroid from a different,
    randomly-chosen community; missing_modality_rate fraction lose one
    modality), except text and image centroid separation are independent
    parameters. overlap_rate is fixed at 0 (hard ground-truth partition;
    this sweep is about agreement/missingness, not overlap)."""
    rng = np.random.default_rng(seed)
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes - sum(sizes)

    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)

    primary = []
    for c, size in enumerate(sizes):
        primary.extend([c] * size)
    primary = np.array(primary[:n_nodes])
    true_communities = [{int(c)} for c in primary]

    text_centroids = rng.normal(scale=text_centroid_scale, size=(n_communities, text_dim))
    image_centroids = rng.normal(scale=image_centroid_scale, size=(n_communities, image_dim))

    e_t = [None] * n_nodes
    e_v = [None] * n_nodes
    misaligned = set()
    for i in range(n_nodes):
        c = int(primary[i])
        e_t[i] = text_centroids[c] + rng.normal(scale=noise_scale, size=text_dim)

        if rng.random() < misalignment_rate:
            wrong_c = int(rng.integers(0, n_communities))
            while wrong_c == c and n_communities > 1:
                wrong_c = int(rng.integers(0, n_communities))
            e_v[i] = image_centroids[wrong_c] + rng.normal(scale=noise_scale, size=image_dim)
            misaligned.add(i)
        else:
            e_v[i] = image_centroids[c] + rng.normal(scale=noise_scale, size=image_dim)

        if rng.random() < missing_modality_rate:
            if rng.random() < 0.5:
                e_t[i] = None
            else:
                e_v[i] = None

    return dict(G=G, e_t=e_t, e_v=e_v, true=true_communities, n_communities=n_communities,
                misaligned_nodes=misaligned)


def build_dataset(regime, agreement, missingness, seed):
    cfg = REGIMES[regime]
    misalignment_rate = round(max(0.0, min(1.0, 1.0 - agreement)), 6)
    return generate_factor_graph(
        n_nodes=N_NODES, n_communities=N_COMM,
        p_in=cfg["p_in"], p_out=cfg["p_out"],
        text_centroid_scale=cfg["text_scale"], image_centroid_scale=cfg["image_scale"],
        noise_scale=cfg["noise_scale"],
        missing_modality_rate=missingness, misalignment_rate=misalignment_rate,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def simple_concat(d, seed):
    """Structural embedding and raw (L2-normalised) content concatenated with
    NO fusion mechanism (no CCA alignment, no ANFIS confidence, no per-node
    alpha weighting) -- both blocks contribute at their raw scale. Fuzzy
    c-means (m=1.5, matching every other clustering step in this project) on
    the concatenation. This is deliberately the "naive" baseline the
    reviewer asked to distinguish from spectral+content (which builds a kNN
    content graph fused into the adjacency before spectral clustering)."""
    from nf_mcd import baselines as b
    from nf_mcd import community_detection as cd
    from nf_mcd import topology as topo

    k = d["n_communities"]
    Zs = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
    Zc = b.content_matrix(d)
    if Zc is None:
        Zc = np.zeros((d["G"].number_of_nodes(), 0))
    X = np.hstack([Zs, Zc])
    U = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(X).U
    return b.Result(cd.defuzzify(U), b._node_to_comm_view(U), k, note="concat(structure,content), no fusion")


def run_method(method, d, seed):
    from nf_mcd import baselines as b

    if method == "structure_only":
        return b.nfmcd_structure_only(d, seed)
    if method == "content_only":
        return b.fcm_content(d, seed)
    if method == "simple_concat":
        return simple_concat(d, seed)
    if method == "spectral_content":
        return b.spectral_graph_content(d, seed)
    if method == "nfmcd_alpha_fixed":
        return b.nfmcd_alpha_fixed(d, seed)
    if method == "nfmcd_adaptive":
        return b.nfmcd_full(d, seed)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def all_jobs():
    jobs = []
    for regime in REGIMES:
        for agreement in AGREEMENT_LEVELS:
            for missing in MISSING_LEVELS:
                for seed in SEEDS:
                    for method in METHODS:
                        jobs.append((regime, agreement, missing, seed, method))
    return jobs


def run_one(job):
    regime, agreement, missing, seed, method = job
    t0 = time.monotonic()
    row = dict(regime=regime, agreement=f"{agreement:.2f}", missingness=f"{missing:.2f}",
               seed=seed, method=method, onmi=float("nan"), modularity=float("nan"),
               f1=float("nan"), k_used="", error="")
    try:
        from nf_mcd import baselines as b
        d = build_dataset(regime, agreement, missing, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_method(method, d, seed)
            s = b.score(d, res)
        row.update(onmi=s["onmi"], modularity=s["modularity"], f1=s["f1"], k_used=res.k_used)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def do_run(workers):
    os.makedirs(EXP, exist_ok=True)
    jobs = all_jobs()
    done = {rkey(r) for r in load_results() if not r.get("error")}

    def jkey(j):
        regime, agreement, missing, seed, method = j
        return (regime, f"{agreement:.2f}", f"{missing:.2f}", seed, method)

    pending = [j for j in jobs if jkey(j) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already done; running {len(pending)} with {workers} workers",
          flush=True)
    write_progress(len(jobs) - len(pending), len(jobs), "starting")
    if not pending:
        write_progress(len(jobs), len(jobs), "all done")
        return
    f, w = open_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, j): j for j in pending}
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if row["error"]:
                    n_err += 1
                    if n_err <= 8:
                        print(f"  FAILED {row['regime']}/{row['agreement']}/{row['missingness']}/"
                              f"{row['method']} seed={row['seed']}: {row['error']}", flush=True)
                if n_done % 50 == 0 or n_done == len(jobs):
                    write_progress(n_done, len(jobs), f"running ({time.monotonic() - t0:.0f}s elapsed)")
                    print(f"  {n_done}/{len(jobs)} done, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors, {time.monotonic() - t0:.0f}s)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors, {time.monotonic() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# calibration check (run once before trusting the sweep)
# ---------------------------------------------------------------------------

def do_calibrate():
    """Verify the two regimes actually give the intended asymmetry: at
    agreement=0.9 (near-clean) and missingness=0, structure_only should beat
    content_only by a wide margin in strong_struct_weak_content, and lose by
    a wide margin in weak_struct_strong_content."""
    from nf_mcd import baselines as b
    lines = ["Factor-sweep regime calibration (agreement=0.9, missingness=0.0, 5 seeds)", "=" * 78]
    for regime in REGIMES:
        onmi = {"structure_only": [], "content_only": []}
        mod = {"structure_only": [], "content_only": []}
        for seed in SEEDS:
            d = build_dataset(regime, 0.9, 0.0, seed)
            for method in ("structure_only", "content_only"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = run_method(method, d, seed)
                    s = b.score(d, res)
                onmi[method].append(s["onmi"])
                mod[method].append(s["modularity"])
        lines.append(f"\n{regime}  (cfg={REGIMES[regime]})")
        for method in ("structure_only", "content_only"):
            lines.append(f"  {method:<16} ONMI={np.mean(onmi[method]):.3f}+-{np.std(onmi[method]):.3f}"
                          f"  modularity={np.mean(mod[method]):.3f}+-{np.std(mod[method]):.3f}")
    text = "\n".join(lines) + "\n"
    atomic_write_text(CALIBRATE_LOG, text)
    print(text)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def do_summary():
    rows = [r for r in load_results() if not r.get("error")]
    if not rows:
        print("No results yet.")
        return

    def cell(regime, agreement, missing, method, metric="onmi"):
        vals = [_fnum(r[metric]) for r in rows
                if r["regime"] == regime and abs(_fnum(r["agreement"]) - agreement) < 1e-6
                and abs(_fnum(r["missingness"]) - missing) < 1e-6 and r["method"] == method]
        vals = [v for v in vals if not np.isnan(v)]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)

    L = []
    L.append("Factor-sweep performance surface: LFK ONMI (mean+-sd over seeds) vs. cross-modal "
              "agreement x missingness, at two structure/content regimes.")
    L.append(f"n_nodes={N_NODES}, n_communities={N_COMM}, {len(SEEDS)} seeds/cell. Total finished rows: {len(rows)}")
    L.append("agreement = 1 - misalignment_rate; missingness = fraction of nodes losing one modality (MCAR).")
    L.append("=" * 100)

    for regime in REGIMES:
        L.append(f"\n\n### regime = {regime}  (cfg={REGIMES[regime]})")
        for method in METHODS:
            L.append(f"\n{method}")
            header = "  agreement\\miss " + "".join(f"{m:>14.1f}" for m in MISSING_LEVELS)
            L.append(header)
            for a in AGREEMENT_LEVELS:
                row = f"  {a:>14.1f} "
                for m in MISSING_LEVELS:
                    mu, sd, n = cell(regime, a, m, method)
                    row += f"{'-':>14}" if np.isnan(mu) else f"{mu:>8.3f}+-{sd:<4.2f}"
                L.append(row)

    L.append("\n\n### Who wins each cell (highest mean ONMI), by regime")
    L.append("=" * 100)
    for regime in REGIMES:
        L.append(f"\n{regime}")
        header = "  agreement\\miss " + "".join(f"{m:>14.1f}" for m in MISSING_LEVELS)
        L.append(header)
        for a in AGREEMENT_LEVELS:
            row = f"  {a:>14.1f} "
            for m in MISSING_LEVELS:
                best_method, best_mu = None, -1.0
                for method in METHODS:
                    mu, _, n = cell(regime, a, m, method)
                    if n and mu > best_mu:
                        best_method, best_mu = method, mu
                label = (best_method[:12] if best_method else "-")
                row += f"{label:>14}"
            L.append(row)

    L.append("\n\n### Modularity surface for nfmcd_adaptive vs nfmcd_alpha_fixed (does adaptive alpha earn its keep?)")
    L.append("=" * 100)
    for regime in REGIMES:
        L.append(f"\n{regime}: mean ONMI(adaptive) - mean ONMI(fixed-0.5), positive = adaptive wins")
        header = "  agreement\\miss " + "".join(f"{m:>10.1f}" for m in MISSING_LEVELS)
        L.append(header)
        for a in AGREEMENT_LEVELS:
            row = f"  {a:>14.1f} "
            for m in MISSING_LEVELS:
                mu_a, _, na = cell(regime, a, m, "nfmcd_adaptive")
                mu_f, _, nf = cell(regime, a, m, "nfmcd_alpha_fixed")
                diff = mu_a - mu_f if (na and nf) else float("nan")
                row += f"{'-':>10}" if np.isnan(diff) else f"{diff:>+10.3f}"
            L.append(row)

    text = "\n".join(L) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)
    make_plots(rows)


def make_plots(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(plotting skipped: {exc})")
        return

    def cell(regime, agreement, missing, method):
        vals = [_fnum(r["onmi"]) for r in rows
                if r["regime"] == regime and abs(_fnum(r["agreement"]) - agreement) < 1e-6
                and abs(_fnum(r["missingness"]) - missing) < 1e-6 and r["method"] == method]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    fig, axes = plt.subplots(2, len(METHODS), figsize=(3.1 * len(METHODS), 6.2))
    for ri, regime in enumerate(REGIMES):
        for mi, method in enumerate(METHODS):
            ax = axes[ri, mi]
            grid = np.array([[cell(regime, a, m, method) for m in MISSING_LEVELS] for a in AGREEMENT_LEVELS])
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="upper")
            ax.set_xticks(range(len(MISSING_LEVELS)))
            ax.set_xticklabels([f"{m:.1f}" for m in MISSING_LEVELS], fontsize=6)
            ax.set_yticks(range(len(AGREEMENT_LEVELS)))
            ax.set_yticklabels([f"{a:.1f}" for a in AGREEMENT_LEVELS], fontsize=6)
            if ri == 0:
                ax.set_title(method, fontsize=8)
            if mi == 0:
                ax.set_ylabel(f"{regime}\nagreement", fontsize=7)
            if ri == 1:
                ax.set_xlabel("missingness", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="mean LFK ONMI")
    fig.suptitle(f"Performance surface: ONMI vs. cross-modal agreement x missingness (n={N_NODES}, k={N_COMM}, "
                 f"{len(SEEDS)} seeds/cell)")
    out = os.path.join(EXP, "factor_sweep_surface.png")
    fig.savefig(out + ".tmp.png", dpi=140)
    plt.close(fig)
    os.replace(out + ".tmp.png", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "run", "summary"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if args.cmd == "calibrate":
        do_calibrate()
    elif args.cmd == "run":
        do_run(args.workers)
        do_summary()
    else:
        do_summary()
