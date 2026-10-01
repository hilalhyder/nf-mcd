"""
CESNA baseline (Yang, McAuley & Leskovec, ICDM 2013: "Community Detection in
Networks with Node Attributes"), added after reviewer feedback that it is "a
natural baseline" for this paper's setting -- overlapping community detection
with node attributes -- and was not cited or compared against.

Model (joint generative model over edges and binary attributes, fit by
gradient ascent -- no labels used anywhere):
  F in R^{n x k}_{>=0}:        per-node community-affiliation strengths
  W in R^{k x d}:               community -> binary-attribute logistic weights
  P(edge u,v)   = 1 - exp(-F_u . F_v)                      (BigCLAM edge model)
  P(attr_u,j=1) = sigmoid(F_u . W_:,j)                      (logistic attribute model)
  L(F,W) = L_network(F) + delta * L_attribute(F,W)

delta balances the two terms; following the original paper's heuristic we set
delta = n_edges / (n*d) so neither likelihood dominates purely from having more
terms. Optimized by full-batch gradient ascent with gradient clipping (F
projected to be non-negative after each step).

Node attributes (CESNA requires BINARY attributes, unlike this paper's own
continuous multimodal embeddings): PCA-reduce the same raw content features
run_gnn_baseline.py/run_vgae_baseline.py use (content_matrix, or the spectral
structural embedding for DBLP/Amazon, which carry no content) to 32
dimensions, then binarize each dimension at its own median. This is a real
methodological choice (not what CESNA's original binary-attribute use case
assumes) and is reported as such rather than silently treated as equivalent
to a native binary-attribute dataset.

Community memberships -> overlapping view: BigCLAM/CESNA's standard threshold
delta_c = sqrt(-ln(1 - eps)), eps = 1/(n*k), applied per community column of F
(Yang & Leskovec 2013's own thresholding rule, not a choice specific to this
baseline's integration here).

Usage (from nfmcd_impl/):
    py run_cesna_baseline.py run
    py run_cesna_baseline.py summary

Crash-safe / resumable: each finished job is appended to
experiments/cesna_baseline_results.csv and fsynced; a rerun skips finished jobs.
"""
from __future__ import annotations

import os

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
RESULTS_CSV = os.path.join(EXP, "cesna_baseline_results.csv")
SUMMARY_LOG = os.path.join(EXP, "cesna_baseline_summary.log")
PROGRESS_MD = os.path.join(EXP, "cesna_baseline_progress.md")

DATASETS = ["crisismmd", "pheme", "fakeddit", "dblp", "amazon"]
SEEDS = (0, 1, 2)
ATTR_DIM = 32
N_ITER = 300
LR = 0.0025
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
        "# CESNA baseline progress\n\n"
        f"- Jobs done: **{done}/{total}**\n"
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- {note}\n\n"
        "## Resume (finished jobs are skipped automatically)\n\n"
        "```\n"
        f"cd {HERE}\n"
        "py run_cesna_baseline.py run\n"
        "py run_cesna_baseline.py summary\n"
        "```\n\n"
        "Results: experiments/cesna_baseline_results.csv (append-only, fsynced per job). "
        "Summary: experiments/cesna_baseline_summary.log.\n"
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
# CESNA (Yang, McAuley & Leskovec 2013): BigCLAM edge model + logistic
# attribute model, joint gradient ascent. Dense O(n^2); fine up to n<=2000.
# ---------------------------------------------------------------------------

def node_features_binary(d: dict, seed: int, attr_dim: int = ATTR_DIM):
    from nf_mcd.baselines import content_matrix
    from nf_mcd import topology as topo
    from sklearn.decomposition import PCA

    X = content_matrix(d)
    if X is not None and X.shape[1] > 0 and np.linalg.norm(X) > 0:
        source = "content_matrix (raw concatenated text+image)"
    else:
        k = d["n_communities"]
        X = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
        source = f"spectral_structural_embedding(dim={k}) -- dataset ships no per-node content"

    dim = min(attr_dim, X.shape[1], X.shape[0] - 1)
    rng_state = np.random.RandomState(seed)
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=dim, random_state=rng_state)
    Z = pca.fit_transform(Xc)
    med = np.median(Z, axis=0, keepdims=True)
    X_bin = (Z > med).astype(np.float64)
    return X_bin, f"{source}, PCA-reduced to {dim} dims, median-binarized per dimension"


def fit_cesna(A: np.ndarray, X_bin: np.ndarray, k: int, seed: int,
              n_iter: int = N_ITER, lr: float = LR):
    n = A.shape[0]
    d = X_bin.shape[1]
    rng = np.random.default_rng(seed)
    F = rng.uniform(0.05, 0.3, size=(n, k))
    W = rng.normal(0, 0.05, size=(k, d))

    n_edges = A.sum() / 2.0
    delta = max(n_edges / max(n * d, 1.0), 1e-4)

    for _ in range(n_iter):
        FFt = np.clip(F @ F.T, 1e-10, 50.0)
        np.fill_diagonal(FFt, 0.0)
        exp_neg = np.exp(-FFt)
        coef_edge = A * (exp_neg / np.clip(1.0 - exp_neg, 1e-10, None))
        coef_nonedge = (1.0 - A) * (-1.0)
        np.fill_diagonal(coef_edge, 0.0)
        np.fill_diagonal(coef_nonedge, 0.0)
        grad_net = (coef_edge + coef_nonedge) @ F

        logits = np.clip(F @ W, -30, 30)
        P = 1.0 / (1.0 + np.exp(-logits))
        err = X_bin - P
        grad_F_attr = err @ W.T
        grad_W_attr = (F.T @ err)

        grad_F = grad_net + delta * grad_F_attr
        gn = np.linalg.norm(grad_F)
        if gn > 1e3:
            grad_F = grad_F * (1e3 / gn)
        F = np.clip(F + lr * grad_F, 1e-6, None)

        gw = np.linalg.norm(grad_W_attr)
        if gw > 1e3:
            grad_W_attr = grad_W_attr * (1e3 / gw)
        W = W + lr * delta * grad_W_attr

        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=1e-6, posinf=10.0, neginf=1e-6)
            F = np.clip(F, 1e-6, 10.0)

    return F, W


def cesna_view(F: np.ndarray):
    n, k = F.shape
    eps = 1.0 / max(n * k, 2)
    delta_c = np.sqrt(-np.log(1 - eps))
    view = [set() for _ in range(k)]
    for u in range(n):
        for c in range(k):
            if F[u, c] > delta_c:
                view[c].add(u)
    for c in range(k):
        if not view[c]:
            view[c].add(int(np.argmax(F[:, c])))
    return view


def true_view(d: dict):
    k = d["n_communities"]
    view = [set() for _ in range(k)]
    for i, comms in enumerate(d["true"]):
        for c in comms:
            if 0 <= c < k:
                view[c].add(i)
    return view


def hard_from_F(F: np.ndarray):
    return np.argmax(F, axis=1)


def run_one(dataset: str, seed: int) -> dict:
    t0 = time.monotonic()
    row = dict(method="cesna", dataset=dataset, seed=seed, k_used="", n_pred="",
               modularity=float("nan"), onmi=float("nan"), f1=float("nan"),
               feature_source="", note="", error="")
    try:
        from nf_mcd import metrics as mx
        import networkx as nx

        with open(os.path.join(CACHE, f"{dataset}.pkl"), "rb") as f:
            d = pickle.load(f)
        G = d["G"]
        n = G.number_of_nodes()
        k = d["n_communities"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), weight=None,
                                          format="csr").astype(np.float64).toarray()
            X_bin, feat_note = node_features_binary(d, seed)

            F, W = fit_cesna(A, X_bin, k, seed)
            hard = hard_from_F(F)
            view = cesna_view(F)
            tview = true_view(d)
            s = dict(
                modularity=float(mx.modularity_score(G, hard)),
                onmi=float(mx.overlapping_nmi(view, tview, n)),
                f1=float(mx.membership_f1(view, tview)),
            )
        row.update(k_used=k, n_pred=len(view), feature_source=feat_note,
                   note=f"CESNA (BigCLAM edge model + logistic attribute model), {N_ITER} iters, "
                        f"gradient ascent", **s)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    row["secs"] = round(time.monotonic() - t0, 2)
    return row


def all_jobs():
    return [(ds, s) for ds in DATASETS for s in SEEDS]


def do_run():
    jobs = all_jobs()
    done = load_results()
    pending = [j for j in jobs if ("cesna", j[0], j[1]) not in done]
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
                      f"f1={row['f1']:.3f} n_pred={row['n_pred']} ({row['secs']:.2f}s)", flush=True)
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
    lines.append("CESNA baseline (Yang, McAuley & Leskovec 2013): BigCLAM edge model + logistic")
    lines.append("attribute model, joint gradient ascent, 300 iterations. k = ground-truth community count.")
    lines.append("Mean±sd over seeds 0,1,2. Scored identically to every other Table 2 baseline:")
    lines.append("modularity (argmax-F hard partition), LFK overlapping NMI (cdlib), membership F1.")
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
    lines.append("Node attribute source per dataset:")
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
