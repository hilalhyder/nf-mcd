"""
Reviewer-requested synthetic-vs-real structural comparison (Experiment 2):
"The scale experiments use the synthetic generator. This demonstrates
computational scalability, but not necessarily real-world scalability...
compare synthetic and real graphs using: degree distribution, clustering
coefficient, community-size distribution, modularity, overlap distribution,
content-similarity distribution. Otherwise the 20,000-node result should be
described specifically as a computational stress test rather than as
evidence of real social-network scalability."

Computes six structural/content statistics for each of the five real
datasets used throughout this project (from experiments/cache/*.pkl, the
exact same cached objects run_baselines.py scores) AND for the synthetic
generator at two configurations:
  - "synthetic_typical": nf_mcd.datasets.generate_synthetic_multimodal_graph
    with its OWN defaults (n_nodes=120, n_communities=4, p_in=0.18,
    p_out=0.02, missing_modality_rate=0.15, misalignment_rate=0.15,
    overlap_rate=0.10) -- exactly the configuration demo.py, README.md's
    usage example, and validate_onmi.py all use as "the" synthetic setting.
  - "synthetic_stress20k": the LITERAL configuration run_scale.py's Part A
    used to produce its headline 20,000-node result -- n_nodes=20000,
    n_communities=8 (run_scale.N_COMMUNITIES), p_in/p_out from
    run_scale.p_in_out_for(20000) (target average degree ~18, same
    p_out/p_in ratio as the generator's own default), and
    missing_modality_rate=0.0, misalignment_rate=0.0, overlap_rate=0.0 (run_scale.py
    deliberately zeroes these out for its clean scaling measurement -- see
    run_scale.py's `_make_dataset`). run_scale.py itself is only imported
    from (for `p_in_out_for` and `N_COMMUNITIES`), never modified.

The six statistics, computed identically across all seven graphs:
  1. Degree distribution: mean, median, max, p99, and tail_ratio = max/p99.
  2. Clustering coefficient: networkx.average_clustering(G).
  3. Community-size distribution (ground truth): n, mean, median, min, max.
  4. Modularity of the ground-truth partition (nf_mcd.metrics.modularity_score,
     using each node's PRIMARY/first-listed community as its hard label when
     a dataset's ground truth allows overlap -- noted explicitly per row).
  5. Overlap fraction: share of nodes belonging to >1 ground-truth community.
  6. Content-similarity distribution: mean cosine similarity of
     nf_mcd.baselines.content_matrix() vectors, within- vs. between-primary-
     community, on a random sample of up to 1500 nodes (full pairwise for
     smaller datasets) -- "not computable" for the two pure-topology SNAP
     datasets (DBLP/Amazon ship no per-node content at all).

Usage (from nfmcd_impl/):
    py run_synth_vs_real.py run        # compute missing rows (resumable)
    py run_synth_vs_real.py summary    # rebuild experiments/synth_vs_real_summary.log

Crash-safe / resumable: each finished dataset row is appended to
experiments/synth_vs_real_results.csv and fsynced; a rerun skips finished rows.
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

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "synth_vs_real_results.csv")
SUMMARY_LOG = os.path.join(EXP, "synth_vs_real_summary.log")
PROGRESS_MD = os.path.join(EXP, "synth_vs_real_progress.md")

REAL_DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
SYNTH_ROWS = ["synthetic_typical", "synthetic_stress20k"]
ALL_ROWS = REAL_DATASETS + SYNTH_ROWS
CONTENT_SAMPLE_CAP = 1500

FIELDS = [
    "dataset", "n", "n_edges",
    "deg_mean", "deg_median", "deg_max", "deg_p99", "deg_tail_ratio",
    "avg_clustering",
    "n_true_communities", "comm_size_mean", "comm_size_median", "comm_size_min", "comm_size_max",
    "overlap_frac", "modularity_gt_primary",
    "content_n_sampled", "content_within_mean", "content_between_mean", "content_delta",
    "note", "error", "secs",
]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# Synthetic-vs-real progress\n\n"
        f"- Rows done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished rows are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_synth_vs_real.py run\n"
        "py run_synth_vs_real.py summary\n"
        "```\n\n"
        "Results: experiments/synth_vs_real_results.csv (append-only, fsynced per row). "
        "Summary: experiments/synth_vs_real_summary.log.\n"
    ))


def load_results():
    res = {}
    if not os.path.exists(RESULTS_CSV):
        return res
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            res[r["dataset"]] = r
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
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return f, w


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def degree_stats(G: nx.Graph) -> dict:
    degs = np.array([d for _, d in G.degree()], dtype=float)
    p99 = float(np.percentile(degs, 99))
    return dict(
        deg_mean=float(degs.mean()), deg_median=float(np.median(degs)),
        deg_max=int(degs.max()), deg_p99=p99,
        deg_tail_ratio=(float(degs.max()) / p99) if p99 > 0 else float("nan"),
    )


def community_size_stats(view) -> dict:
    sizes = np.array([len(c) for c in view if len(c) > 0], dtype=float)
    if len(sizes) == 0:
        return dict(n_true_communities=0, comm_size_mean=float("nan"), comm_size_median=float("nan"),
                    comm_size_min=float("nan"), comm_size_max=float("nan"))
    return dict(
        n_true_communities=int(len(sizes)), comm_size_mean=float(sizes.mean()),
        comm_size_median=float(np.median(sizes)), comm_size_min=float(sizes.min()), comm_size_max=float(sizes.max()),
    )


def primary_labels(true_per_node) -> np.ndarray:
    return np.array([min(c) if c else -1 for c in true_per_node])


def overlap_fraction(true_per_node) -> float:
    n = len(true_per_node)
    if n == 0:
        return float("nan")
    return sum(1 for c in true_per_node if len(c) > 1) / n


def community_view(true_per_node, k) -> list:
    view = [set() for _ in range(k)]
    for i, comms in enumerate(true_per_node):
        for c in comms:
            if 0 <= c < k:
                view[c].add(i)
    return view


def modularity_of_ground_truth(G: nx.Graph, true_per_node) -> float:
    from nf_mcd import metrics as mx
    hard = primary_labels(true_per_node)
    return float(mx.modularity_score(G, hard))


def content_similarity_stats(G: nx.Graph, e_t, e_v, true_per_node, seed: int = 0,
                              sample_cap: int = CONTENT_SAMPLE_CAP) -> dict:
    from nf_mcd.baselines import content_matrix

    d = dict(G=G, e_t=e_t, e_v=e_v)
    X = content_matrix(d)
    if X is None:
        return dict(content_n_sampled=0, content_within_mean=float("nan"),
                     content_between_mean=float("nan"), content_delta=float("nan"),
                     note_content="no per-node content in this dataset")
    n = X.shape[0]
    norms_full = np.linalg.norm(X, axis=1)
    has_content = np.where(norms_full > 1e-9)[0]
    rng = np.random.default_rng(seed)
    take = min(sample_cap, len(has_content))
    idx = rng.choice(has_content, size=take, replace=False) if take > 0 else has_content
    Xs = X[idx]
    norms = np.linalg.norm(Xs, axis=1, keepdims=True)
    Xn = Xs / (norms + 1e-12)
    S = Xn @ Xn.T
    primary = np.array([min(true_per_node[i]) if true_per_node[i] else -1 for i in idx])
    n_s = len(idx)
    iu = np.triu_indices(n_s, k=1)
    same = primary[iu[0]] == primary[iu[1]]
    within = S[iu][same]
    between = S[iu][~same]
    wm = float(within.mean()) if len(within) else float("nan")
    bm = float(between.mean()) if len(between) else float("nan")
    return dict(content_n_sampled=n_s, content_within_mean=wm, content_between_mean=bm,
                content_delta=(wm - bm) if len(within) and len(between) else float("nan"), note_content="")


def _make_real_row(dataset: str) -> dict:
    with open(os.path.join(CACHE, f"{dataset}.pkl"), "rb") as f:
        d = pickle.load(f)
    G, true_per_node, k = d["G"], d["true"], d["n_communities"]
    return _make_row(dataset, G, true_per_node, k, d["e_t"], d["e_v"], seed=0,
                      note="real dataset (cached, same object run_baselines.py scores)")


def _make_synth_row(name: str) -> dict:
    from nf_mcd.datasets import generate_synthetic_multimodal_graph

    if name == "synthetic_typical":
        data = generate_synthetic_multimodal_graph(n_nodes=120, n_communities=4, seed=42)
        note = ("synthetic generator's OWN defaults (n_nodes=120, n_communities=4, p_in=0.18, p_out=0.02, "
                "missing_modality_rate=0.15, misalignment_rate=0.15, overlap_rate=0.10) -- "
                "the configuration demo.py/README.md/validate_onmi.py all use")
    elif name == "synthetic_stress20k":
        from run_scale import p_in_out_for, N_COMMUNITIES
        p_in, p_out = p_in_out_for(20000)
        data = generate_synthetic_multimodal_graph(
            n_nodes=20000, n_communities=N_COMMUNITIES, p_in=p_in, p_out=p_out,
            missing_modality_rate=0.0, misalignment_rate=0.0, overlap_rate=0.0, seed=0,
        )
        note = (f"LITERAL run_scale.py Part A config for its 20,000-node headline result: n_communities="
                f"{N_COMMUNITIES}, p_in={p_in:.5f}, p_out={p_out:.5f} (target avg degree ~18), "
                f"missing/misalignment/overlap all forced to 0")
    else:
        raise ValueError(name)
    return _make_row(name, data.G, data.true_communities, data.n_communities,
                      data.text_embeddings, data.image_embeddings, seed=0, note=note)


def _make_row(name, G, true_per_node, k, e_t, e_v, seed, note) -> dict:
    t0 = time.monotonic()
    row = {f: "" for f in FIELDS}
    row["dataset"] = name
    try:
        n = G.number_of_nodes()
        row["n"] = n
        row["n_edges"] = G.number_of_edges()
        row.update(degree_stats(G))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["avg_clustering"] = float(nx.average_clustering(G))
        view = community_view(true_per_node, k)
        row.update(community_size_stats(view))
        row["overlap_frac"] = overlap_fraction(true_per_node)
        row["modularity_gt_primary"] = modularity_of_ground_truth(G, true_per_node)
        cs = content_similarity_stats(G, e_t, e_v, true_per_node, seed=seed)
        note_content = cs.pop("note_content", "")
        row.update(cs)
        row["note"] = note + (f"; content: {note_content}" if note_content else "")
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def do_run():
    done = load_results()
    pending = [ds for ds in ALL_ROWS if ds not in done]
    print(f"{len(ALL_ROWS) - len(pending)}/{len(ALL_ROWS)} rows already in CSV; computing {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(ALL_ROWS) - len(pending)
    n_err = 0
    write_progress(n_done, len(ALL_ROWS), "running")
    try:
        for ds in pending:
            print(f"  computing {ds} ...", flush=True)
            row = _make_row_dispatch(ds)
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            n_done += 1
            if row["error"]:
                n_err += 1
                print(f"  FAILED {ds}: {row['error']}", flush=True)
            else:
                print(f"  ok {ds}: n={row['n']} edges={row['n_edges']} "
                      f"deg_mean={row['deg_mean']:.2f} clustering={row['avg_clustering']:.4f} "
                      f"({row['secs']:.1f}s)", flush=True)
            write_progress(n_done, len(ALL_ROWS), "running")
    finally:
        f.close()
    write_progress(n_done, len(ALL_ROWS), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(ALL_ROWS)} rows, {n_err} errors", flush=True)


def _make_row_dispatch(ds):
    if ds in REAL_DATASETS:
        return _make_real_row(ds)
    return _make_synth_row(ds)


def do_summary():
    rows = load_results()
    lines = []
    lines.append("Synthetic-vs-real structural comparison (reviewer-requested calibration of the")
    lines.append("20,000-node scale claim). 5 real datasets (cached objects run_baselines.py scores)")
    lines.append("+ 2 synthetic configurations: 'synthetic_typical' (generator defaults, matches")
    lines.append("demo.py/README) and 'synthetic_stress20k' (literal run_scale.py Part A 20k-node config).")
    lines.append("")

    def g(ds, k, fmt="{:.3f}"):
        r = rows.get(ds)
        if r is None:
            return "missing"
        v = r.get(k, "")
        if v in ("", None):
            return "n/a"
        try:
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return str(v)

    header = f"{'dataset':<20}" + "".join(f"{ds:>14}" for ds in ALL_ROWS)
    lines.append("1) Degree distribution")
    lines.append("-" * len(header))
    lines.append(header)
    for k, label in (("n", "n_nodes"), ("n_edges", "n_edges"), ("deg_mean", "mean"), ("deg_median", "median"),
                     ("deg_max", "max"), ("deg_p99", "p99"), ("deg_tail_ratio", "max/p99")):
        fmt = "{:.0f}" if k in ("n", "n_edges", "deg_max") else "{:.3f}"
        lines.append(f"{label:<20}" + "".join(f"{g(ds, k, fmt):>14}" for ds in ALL_ROWS))

    lines.append("\n2) Clustering coefficient (nx.average_clustering)")
    lines.append("-" * len(header))
    lines.append(header)
    lines.append(f"{'avg_clustering':<20}" + "".join(f"{g(ds, 'avg_clustering'):>14}" for ds in ALL_ROWS))

    lines.append("\n3) Community-size distribution (ground truth)")
    lines.append("-" * len(header))
    lines.append(header)
    for k, label in (("n_true_communities", "n_communities"), ("comm_size_mean", "mean"),
                     ("comm_size_median", "median"), ("comm_size_min", "min"), ("comm_size_max", "max")):
        fmt = "{:.0f}" if k != "comm_size_mean" else "{:.1f}"
        lines.append(f"{label:<20}" + "".join(f"{g(ds, k, fmt):>14}" for ds in ALL_ROWS))

    lines.append("\n4) Modularity of ground-truth partition (primary/first community per node)")
    lines.append("-" * len(header))
    lines.append(header)
    lines.append(f"{'modularity_gt':<20}" + "".join(f"{g(ds, 'modularity_gt_primary'):>14}" for ds in ALL_ROWS))

    lines.append("\n5) Overlap distribution (fraction of nodes in >1 ground-truth community)")
    lines.append("-" * len(header))
    lines.append(header)
    lines.append(f"{'overlap_frac':<20}" + "".join(f"{g(ds, 'overlap_frac'):>14}" for ds in ALL_ROWS))

    lines.append("\n6) Content-similarity distribution (cosine sim, sampled <=1500 nodes; 'n/a' = no content)")
    lines.append("-" * len(header))
    lines.append(header)
    for k, label in (("content_n_sampled", "n_sampled"), ("content_within_mean", "within-comm mean"),
                     ("content_between_mean", "between-comm mean"), ("content_delta", "within-between delta")):
        fmt = "{:.0f}" if k == "content_n_sampled" else "{:.4f}"
        lines.append(f"{label:<20}" + "".join(f"{g(ds, k, fmt):>14}" for ds in ALL_ROWS))

    lines.append("\nNotes per row:")
    for ds in ALL_ROWS:
        r = rows.get(ds)
        if r:
            lines.append(f"  {ds}: {r.get('note', '')}")

    text = "\n".join(lines) + "\n"
    atomic_write_text(SUMMARY_LOG, text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    args = ap.parse_args()
    if args.cmd == "run":
        do_run()
        do_summary()
    else:
        do_summary()
