"""
Reviewer-requested GNN/attributed-graph baseline (Experiment 1 of the
reviewer's requested "engage explicitly with GNN/graph-autoencoder related
work" revision).

torch is NOT installed in this environment (`py -c "import torch"` fails
despite the manuscript's reproducibility section claiming "torch 2.14.0 (CPU
build)"), and neither is torch_geometric. Per the task instructions, no new
large dependency was pip-installed. Instead this script hand-rolls a minimal,
honest 2-layer Graph Convolutional Network (Kipf & Welling 2017 propagation
rule: symmetric-normalized adjacency with self-loops) purely in numpy:

    H1 = ReLU(A_hat @ X  @ W0)
    H2 =      A_hat @ H1 @ W1

W0/W1 are randomly initialised (Glorot/Xavier scale) and NEVER TRAINED -- no
gradient descent, no autoencoder reconstruction loss, no labels used anywhere
in the forward pass. This is the "untrained GCN as a smoothing/embedding
operator" baseline explicitly sanctioned as legitimate by the task brief,
closely related to SGC (Simplifying Graph Convolutional Networks, Wu et al.
2019, ICML) which shows a linear/untrained propagation operator over the
normalized adjacency is a strong, citable lightweight stand-in for a full
trained GNN. Do not describe this in the manuscript as a trained GNN or as a
graph autoencoder -- it is neither.

Node features X:
  - crisismmd, pheme, fakeddit: nf_mcd.baselines.content_matrix(d) -- the
    SAME raw (not NF-MCD-fused) concatenated, L2-normalized text/image
    embeddings that the project's own kmeans_content/fcm_content baselines
    already use. A missing modality contributes a zero block (pheme has no
    images at all: e_v block is all zero).
  - dblp, amazon: these SNAP datasets ship literally no per-node content
    (nf_mcd/datasets.py: "texts/images are [None]*n by construction"), so
    content_matrix() returns None. For these two only, X is
    nf_mcd.topology.compute_structural_embedding(G, dim=n_communities) --
    the exact same spectral embedding the project's own graph-only ablations
    (nfmcd_structure_only, fcm_structure_only) already fall back to for these
    datasets, so the GCN baseline is not disadvantaged by an arbitrary
    feature choice not otherwise used in this codebase.

The resulting node embeddings (L2-row-normalized) are clustered with this
project's own FuzzyCMeans(m=1.5) (nf_mcd.community_detection), at k = the
ground-truth community count (same protocol as the rest of Table 2), and
scored identically to every other baseline in run_baselines.py: modularity
of the hard (argmax) partition, LFK overlapping NMI (cdlib) at overlap
threshold 0.2, and best-match membership F1 (nf_mcd/metrics.py).

Usage (from nfmcd_impl/):
    py run_gnn_baseline.py run        # run missing (dataset, seed) jobs
    py run_gnn_baseline.py summary    # rebuild experiments/gnn_baseline_summary.log

Crash-safe / resumable: each finished job is appended to
experiments/gnn_baseline_results.csv and fsynced; a rerun skips finished jobs.
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "gnn_baseline_results.csv")
SUMMARY_LOG = os.path.join(EXP, "gnn_baseline_summary.log")
PROGRESS_MD = os.path.join(EXP, "gnn_baseline_progress.md")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
SEEDS = (0, 1, 2)
OVERLAP_THRESHOLD = 0.2
HIDDEN_DIM = 64
OUT_DIM = 32
FIELDS = ["method", "dataset", "seed", "k_used", "n_pred", "modularity", "onmi", "f1",
          "feature_source", "note", "error", "secs"]


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_progress(done, total, note=""):
    atomic_write_text(PROGRESS_MD, (
        "# GNN baseline progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_gnn_baseline.py run\n"
        "py run_gnn_baseline.py summary\n"
        "```\n\n"
        "Results: experiments/gnn_baseline_results.csv (append-only, fsynced per job). "
        "Summary: experiments/gnn_baseline_summary.log.\n"
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
                for k in ("modularity", "onmi", "f1"):
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
# Hand-rolled, untrained 2-layer GCN (numpy only; no torch available)
# ---------------------------------------------------------------------------

def normalized_adj(G, n: int) -> np.ndarray:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2
    (Kipf & Welling 2017 propagation rule). Dense: fine up to this project's
    dataset sizes (n <= 2000)."""
    import networkx as nx

    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), weight=None, format="csr").astype(float).toarray()
    A = A + np.eye(n)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def gcn_embed(A_hat: np.ndarray, X: np.ndarray, seed: int,
              hidden: int = HIDDEN_DIM, out: int = OUT_DIM) -> np.ndarray:
    """2-layer GCN forward pass with random, FROZEN (untrained) weights:
    H1 = ReLU(A_hat @ X @ W0); H2 = A_hat @ H1 @ W1. Glorot-scaled init."""
    rng = np.random.default_rng(seed)
    d_in = X.shape[1]
    W0 = rng.normal(scale=np.sqrt(2.0 / (d_in + hidden)), size=(d_in, hidden))
    H1 = np.maximum(A_hat @ (X @ W0), 0.0)
    W1 = rng.normal(scale=np.sqrt(2.0 / (hidden + out)), size=(hidden, out))
    H2 = A_hat @ (H1 @ W1)
    norms = np.linalg.norm(H2, axis=1, keepdims=True)
    return H2 / (norms + 1e-12)


def node_features(d: dict, seed: int):
    from nf_mcd.baselines import content_matrix
    from nf_mcd import topology as topo

    X = content_matrix(d)
    if X is not None and X.shape[1] > 0 and np.linalg.norm(X) > 0:
        return X, "content_matrix (raw concatenated L2-normalized text+image, zero-block if a modality is missing)"
    k = d["n_communities"]
    Z = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
    return Z, f"spectral_structural_embedding(dim={k}) -- dataset ships no per-node content"


def true_view(d: dict):
    k = d["n_communities"]
    view = [set() for _ in range(k)]
    for i, comms in enumerate(d["true"]):
        for c in comms:
            if 0 <= c < k:
                view[c].add(i)
    return view


def node_to_comm_view(U: np.ndarray):
    from nf_mcd import community_detection as cd
    view = [set() for _ in range(U.shape[1])]
    for i, comms in enumerate(cd.overlapping_communities(U, OVERLAP_THRESHOLD)):
        for c in comms:
            view[c].add(i)
    return view


def run_one(dataset: str, seed: int) -> dict:
    t0 = time.monotonic()
    row = dict(method="gcn_embed_fcm", dataset=dataset, seed=seed, k_used="", n_pred="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"),
               feature_source="", note="", error="")
    try:
        from nf_mcd import community_detection as cd
        from nf_mcd import metrics as mx

        with open(os.path.join(CACHE, f"{dataset}.pkl"), "rb") as f:
            d = pickle.load(f)
        G = d["G"]
        n = G.number_of_nodes()
        k = d["n_communities"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_hat = normalized_adj(G, n)
            X, feat_note = node_features(d, seed)
            H = gcn_embed(A_hat, X, seed)
            fcm = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(H)
            U = fcm.U
            hard = cd.defuzzify(U)
            view = node_to_comm_view(U)
            tview = true_view(d)
            s = dict(
                modularity=float(mx.modularity_score(G, hard)),
                onmi=float(mx.overlapping_nmi(view, tview, n)),
                f1=float(mx.membership_f1(view, tview)),
            )
        row.update(k_used=k, n_pred=len(view), feature_source=feat_note,
                   note=f"untrained 2-layer GCN (hidden={HIDDEN_DIM},out={OUT_DIM}) + FuzzyCMeans(m=1.5)", **s)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    return [(ds, s) for ds in DATASETS for s in SEEDS]


def do_run():
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if ("gcn_embed_fcm", j[0], j[1]) not in done]
    print(f"{len(jobs) - len(pending)}/{len(jobs)} jobs already in CSV; running {len(pending)}", flush=True)
    if not pending:
        return
    f, w = open_for_append()
    n_done = len(jobs) - len(pending)
    n_err = 0
    write_progress(n_done, len(jobs), "running")
    try:
        for ds, seed in pending:
            row = run_one(ds, seed)
            w.writerow([row[k] for k in FIELDS])
            f.flush()
            os.fsync(f.fileno())
            n_done += 1
            if row["error"]:
                n_err += 1
                print(f"  FAILED {ds}/seed{seed}: {row['error']}", flush=True)
            else:
                print(f"  ok {ds}/seed{seed}: onmi={row['onmi']:.3f} mod={row['modularity']:.3f} "
                      f"f1={row['f1']:.3f} ({row['secs']:.2f}s)", flush=True)
            write_progress(n_done, len(jobs), "running")
    finally:
        f.close()
    write_progress(n_done, len(jobs), f"finished ({n_err} errors)")
    print(f"done: {n_done}/{len(jobs)} jobs, {n_err} errors", flush=True)


def _fmt(vals):
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f}±{a.std(ddof=0):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


def do_summary():
    res = load_results()
    agg = defaultdict(lambda: defaultdict(list))
    feat = {}
    for (m, ds, s), r in res.items():
        for key in ("modularity", "onmi", "f1"):
            agg[ds][key].append(r[key])
        agg[ds]["npred"].append(r["n_pred"])
        feat[ds] = r.get("feature_source", "")

    lines = []
    lines.append("GNN baseline: untrained 2-layer GCN embedding (Kipf-Welling propagation, random frozen")
    lines.append("weights, ReLU) + FuzzyCMeans(m=1.5) clustering. k = ground-truth community count.")
    lines.append("Mean±sd over seeds 0,1,2. Scored identically to experiments/baselines_summary.log:")
    lines.append("modularity (hard partition), LFK overlapping NMI (cdlib, threshold 0.2), membership F1.")
    lines.append("")
    header = f"{'dataset':<12}{'onmi':>16}{'modularity':>16}{'f1':>16}{'n_pred(mean)':>16}"
    lines.append(header)
    for ds in DATASETS:
        if ds not in agg:
            lines.append(f"{ds:<12}{'missing':>16}")
            continue
        npred = np.mean([float(x) for x in agg[ds]["npred"]])
        lines.append(f"{ds:<12}{_fmt(agg[ds]['onmi']):>16}{_fmt(agg[ds]['modularity']):>16}"
                      f"{_fmt(agg[ds]['f1']):>16}{npred:>16.1f}")
    lines.append("")
    lines.append("Node feature source per dataset:")
    for ds in DATASETS:
        if ds in feat:
            lines.append(f"  {ds:<12}{feat[ds]}")

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
