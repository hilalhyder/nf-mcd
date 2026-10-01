"""
Trained graph-autoencoder baseline (VGAE, Kipf & Welling 2016, ref [23] in the
manuscript -- cited but, until this script, never actually run).

Unlike run_gnn_baseline.py's untrained GCN (random frozen weights, no gradient
descent), this is a genuinely trained encoder: a 2-layer GCN produces per-node
(mu, logstd), reparameterized to z, decoded via an inner-product adjacency
reconstruction (sigmoid(z @ z.T)), and trained end-to-end by Adam on a
weighted binary cross-entropy (positive/negative edge class balance) plus a
KL-divergence term against a standard normal prior -- the standard VGAE
objective. No labels are used anywhere in training or the loss.

Requires torch (installed into a separate short-path venv at C:\\pytv to work
around a Windows long-path limit on this machine; see experiments/
vgae_baseline_progress.md for the exact install commands if reproducing).

Node features X and the clustering/scoring protocol are IDENTICAL to
run_gnn_baseline.py for a fair, apples-to-apples comparison:
  - crisismmd, pheme, fakeddit: nf_mcd.baselines.content_matrix(d).
  - dblp, amazon (no native content): nf_mcd.topology.compute_structural_embedding.
Embeddings (mu at inference, L2-row-normalized) are clustered with this
project's own FuzzyCMeans(m=1.5) at k = ground-truth community count, and
scored identically to every other baseline in run_baselines.py: modularity of
the hard (argmax) partition, LFK overlapping NMI (cdlib, threshold 0.2), and
best-match membership F1.

Usage (from nfmcd_impl/, using the torch-enabled venv):
    C:\\pytv\\Scripts\\python.exe run_vgae_baseline.py run
    C:\\pytv\\Scripts\\python.exe run_vgae_baseline.py summary

Crash-safe / resumable: each finished job is appended to
experiments/vgae_baseline_results.csv and fsynced; a rerun skips finished jobs.
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
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
CACHE = os.path.join(EXP, "cache")
RESULTS_CSV = os.path.join(EXP, "vgae_baseline_results.csv")
SUMMARY_LOG = os.path.join(EXP, "vgae_baseline_summary.log")
PROGRESS_MD = os.path.join(EXP, "vgae_baseline_progress.md")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
SEEDS = (0, 1, 2)
OVERLAP_THRESHOLD = 0.2
HIDDEN_DIM = 64
OUT_DIM = 32
EPOCHS = 200
LR = 0.01
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
        "# VGAE baseline progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "C:\\pytv\\Scripts\\python.exe run_vgae_baseline.py run\n"
        "C:\\pytv\\Scripts\\python.exe run_vgae_baseline.py summary\n"
        "```\n\n"
        "Results: experiments/vgae_baseline_results.csv (append-only, fsynced per job). "
        "Summary: experiments/vgae_baseline_summary.log.\n"
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
# VGAE (Kipf & Welling 2016): trained 2-layer GCN encoder, inner-product decoder
# ---------------------------------------------------------------------------

def normalized_adj_torch(G, n: int) -> torch.Tensor:
    import networkx as nx
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), weight=None, format="csr").astype(float).toarray()
    A = A + np.eye(n)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    A_hat = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    return torch.tensor(A_hat, dtype=torch.float32)


class VGAEEncoder(nn.Module):
    def __init__(self, d_in, hidden, out):
        super().__init__()
        self.W0 = nn.Linear(d_in, hidden, bias=False)
        self.W_mu = nn.Linear(hidden, out, bias=False)
        self.W_logstd = nn.Linear(hidden, out, bias=False)

    def forward(self, A_hat, X):
        H1 = torch.relu(A_hat @ self.W0(X))
        mu = A_hat @ self.W_mu(H1)
        logstd = A_hat @ self.W_logstd(H1)
        return mu, logstd


def train_vgae(A_hat: torch.Tensor, X: torch.Tensor, A_target: torch.Tensor, seed: int,
               hidden: int = HIDDEN_DIM, out: int = OUT_DIM, epochs: int = EPOCHS, lr: float = LR) -> np.ndarray:
    torch.manual_seed(seed)
    n = X.shape[0]
    enc = VGAEEncoder(X.shape[1], hidden, out)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)

    n_edges_pos = A_target.sum()
    n_total = n * n
    pos_weight = (n_total - n_edges_pos) / max(n_edges_pos.item(), 1.0)
    norm = n_total / float((n_total - n_edges_pos) * 2)

    for _ in range(epochs):
        opt.zero_grad()
        mu, logstd = enc(A_hat, X)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits = z @ z.T
        recon_loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, A_target, pos_weight=pos_weight)
        kl = -0.5 / n * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1))
        loss = recon_loss + kl
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu, _ = enc(A_hat, X)
    H = mu.numpy()
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / (norms + 1e-12)


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
    row = dict(method="vgae_fcm", dataset=dataset, seed=seed, k_used="", n_pred="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"),
               feature_source="", note="", error="")
    try:
        from nf_mcd import community_detection as cd
        from nf_mcd import metrics as mx
        import networkx as nx

        with open(os.path.join(CACHE, f"{dataset}.pkl"), "rb") as f:
            d = pickle.load(f)
        G = d["G"]
        n = G.number_of_nodes()
        k = d["n_communities"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_hat = normalized_adj_torch(G, n)
            X_np, feat_note = node_features(d, seed)
            X = torch.tensor(X_np, dtype=torch.float32)
            A_dense = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), weight=None,
                                                format="csr").astype(float).toarray()
            np.fill_diagonal(A_dense, 1.0)
            A_target = torch.tensor(A_dense, dtype=torch.float32)

            H = train_vgae(A_hat, X, A_target, seed)
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
                   note=f"trained VGAE (hidden={HIDDEN_DIM},out={OUT_DIM},epochs={EPOCHS}) + FuzzyCMeans(m=1.5)",
                   **s)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    return [(ds, s) for ds in DATASETS for s in SEEDS]


def do_run():
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if ("vgae_fcm", j[0], j[1]) not in done]
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
    return f"{a.mean():.3f}\u00b1{a.std(ddof=0):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


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
    lines.append("VGAE baseline: TRAINED 2-layer GCN encoder (Kipf & Welling 2016), inner-product")
    lines.append(f"decoder, Adam optimizer, {EPOCHS} epochs, weighted BCE + KL loss (standard VGAE objective).")
    lines.append("mu at inference, L2-row-normalized, + FuzzyCMeans(m=1.5) clustering. k = ground-truth count.")
    lines.append("Mean\u00b1sd over seeds 0,1,2. Scored identically to experiments/baselines_summary.log and")
    lines.append("experiments/gnn_baseline_summary.log (the untrained-GCN baseline): modularity (hard partition),")
    lines.append("LFK overlapping NMI (cdlib, threshold 0.2), membership F1.")
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
    lines.append("Node feature source per dataset (identical to the untrained-GCN baseline):")
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
