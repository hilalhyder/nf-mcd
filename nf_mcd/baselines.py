"""
Baselines and ablations for the NF-MCD evaluation grid (paper Section 5.2).

Every method takes a cached dataset dict `d` (keys: G, e_t, e_v, true,
n_communities) and a seed, and returns a `Result` with a hard label vector
(argmax / single community per node) and a per-community node-index view used
for overlapping metrics. `score()` evaluates every method identically:
modularity of the hard partition, LFK overlapping NMI (cdlib) and best-match
membership F1, against the ground-truth communities.

Protocol notes
--------------
* Methods that take a community count use k = number of ground-truth
  communities (`d["n_communities"]`). Louvain, DEMON and SLPA choose their own
  number of communities.
* Fuzzy methods use `overlapping_communities(U, 0.2)`, which always includes
  each node's strongest community. Hard methods put every node in one community.
* For overlapping methods that leave nodes uncovered (DEMON, SLPA), the hard
  partition used for modularity gives each uncovered node its own singleton
  label; for ONMI/F1 uncovered nodes simply belong to no community.
* Ablations are NF-MCD with unchanged defaults, only the inputs are masked
  (or alpha fixed) - no NF-MCD default is modified.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from . import community_detection as cd
from . import metrics as mx
from . import topology as topo
from .pipeline import NFMCD

OVERLAP_THRESHOLD = 0.2
KNN = 10


@dataclass
class Result:
    hard: np.ndarray
    view: List[Set[int]]
    k_used: int
    note: str = ""


def hard_to_view(labels: np.ndarray) -> List[Set[int]]:
    view: Dict[int, Set[int]] = {}
    for i, lab in enumerate(labels):
        view.setdefault(int(lab), set()).add(i)
    return list(view.values())


def _true_view(true_per_node, k_true: int) -> List[Set[int]]:
    view: List[Set[int]] = [set() for _ in range(k_true)]
    for i, comms in enumerate(true_per_node):
        for c in comms:
            if 0 <= c < k_true:
                view[c].add(i)
    return view


def score(d: dict, res: Result) -> Dict[str, float]:
    n = d["G"].number_of_nodes()
    true_view = _true_view(d["true"], d["n_communities"])
    return dict(
        modularity=float(mx.modularity_score(d["G"], res.hard)),
        onmi=float(mx.overlapping_nmi(res.view, true_view, n)),
        f1=float(mx.membership_f1(res.view, true_view)),
        n_pred=len(res.view),
    )


# ---------------------------------------------------------------------------
# NF-MCD and its ablations (defaults untouched; only inputs are masked)
# ---------------------------------------------------------------------------

def _fit_nfmcd(d, k, seed, text=True, image=True, alpha=None) -> NFMCD:
    n = d["G"].number_of_nodes()
    e_t = d["e_t"] if text else [None] * n
    e_v = d["e_v"] if image else [None] * n
    kwargs = {}
    if alpha is not None:
        kwargs.update(alpha_min=alpha, alpha_max=alpha)
    model = NFMCD(n_communities=k, seed=seed, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=e_t, image_embeddings=e_v)
    return model


def _nfmcd_result(model: NFMCD, note: str = "") -> Result:
    return Result(
        hard=model.predict_hard(),
        view=[c for c in model.overlapping_communities_view(OVERLAP_THRESHOLD)],
        k_used=model.n_communities,
        note=note,
    )


def nfmcd_full(d, seed):
    return _nfmcd_result(_fit_nfmcd(d, d["n_communities"], seed))


def nfmcd_scan_k(d, seed):
    """NF-MCD with its own k: best modularity over {k-1, k, k+1} (as in run_real_experiments.py)."""
    k0 = d["n_communities"]
    best = None
    for k in sorted({kk for kk in (k0 - 1, k0, k0 + 1) if kk >= 2}):
        model = _fit_nfmcd(d, k, seed)
        mod = mx.modularity_score(d["G"], model.predict_hard())
        if best is None or mod > best[0]:
            best = (mod, model)
    return _nfmcd_result(best[1], note=f"scan-k={best[1].n_communities}")


def nfmcd_text_only(d, seed):
    return _nfmcd_result(_fit_nfmcd(d, d["n_communities"], seed, image=False))


def nfmcd_image_only(d, seed):
    return _nfmcd_result(_fit_nfmcd(d, d["n_communities"], seed, text=False))


def nfmcd_alpha_fixed(d, seed):
    return _nfmcd_result(_fit_nfmcd(d, d["n_communities"], seed, alpha=0.5))


def nfmcd_structure_only(d, seed):
    return _nfmcd_result(_fit_nfmcd(d, d["n_communities"], seed, text=False, image=False))


@dataclass
class LinearConfidence:
    """Simplest possible alternative to ANFISAgreement: a direct linear
    rescaling of cosine agreement into [0, 1], with no membership functions,
    no rule structure, and no fitted/calibrated parameters. Implements the
    same call interface (`infer`) so it can be dropped into NFMCD's `anfis=`
    slot unchanged. Added to check whether the fuzzy ANFIS layer earns its
    added complexity over the simplest reasonable alternative (reviewer
    request; paper Section 5.7)."""

    def infer(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(s)
        return np.clip((s + 1.0) / 2.0, 0.0, 1.0)


def nfmcd_linear_confidence(d, seed):
    model = NFMCD(n_communities=d["n_communities"], seed=seed, anfis=LinearConfidence())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["G"], text_embeddings=d["e_t"], image_embeddings=d["e_v"])
    return _nfmcd_result(model, note="linear confidence c=(s+1)/2, replaces ANFIS")


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def content_matrix(d) -> Optional[np.ndarray]:
    """Concatenated L2-normalised text and image embeddings; a missing
    modality contributes a zero block. None if the dataset has no content."""
    n = d["G"].number_of_nodes()
    blocks = []
    for key in ("e_t", "e_v"):
        vecs = d[key]
        dim = next((v.shape[0] for v in vecs if v is not None), None)
        if dim is None:
            continue
        M = np.zeros((n, dim))
        for i, v in enumerate(vecs):
            if v is not None:
                M[i] = v / (np.linalg.norm(v) + 1e-12)
        blocks.append(M)
    return np.hstack(blocks) if blocks else None


# ---------------------------------------------------------------------------
# Non-fuzzy baselines
# ---------------------------------------------------------------------------

def kmeans_content(d, seed):
    from sklearn.cluster import KMeans

    X = content_matrix(d)
    labels = KMeans(n_clusters=d["n_communities"], n_init=10, random_state=seed).fit_predict(X)
    return Result(labels, hard_to_view(labels), d["n_communities"])


def louvain(d, seed):
    comms = nx.community.louvain_communities(d["G"], seed=seed)
    labels = np.zeros(d["G"].number_of_nodes(), dtype=int)
    for c, members in enumerate(comms):
        for v in members:
            labels[v] = c
    return Result(labels, hard_to_view(labels), len(comms))


def spectral_graph_content(d, seed):
    """Spectral clustering on adjacency (+ a kNN cosine content graph when the
    dataset has content). With no content this is plain spectral clustering."""
    from sklearn.cluster import SpectralClustering

    G = d["G"]
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=None, format="csr").astype(float).toarray()
    note = "graph-only"
    X = content_matrix(d)
    if X is not None:
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        kn = min(KNN, n - 1)
        idx = np.argpartition(-S, kn - 1, axis=1)[:, :kn]
        K = np.zeros((n, n))
        rows = np.repeat(np.arange(n), kn)
        K[rows, idx.ravel()] = 1.0
        K = np.maximum(K, K.T)
        A = A + K
        note = f"graph+{KNN}NN-content"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = SpectralClustering(
            n_clusters=d["n_communities"], affinity="precomputed",
            random_state=seed, assign_labels="kmeans",
        ).fit_predict(A)
    return Result(labels, hard_to_view(labels), d["n_communities"], note=note)


# ---------------------------------------------------------------------------
# Fuzzy / overlapping topology-only baselines
# ---------------------------------------------------------------------------

def fcm_structure_only(d, seed):
    k = d["n_communities"]
    Z = topo.compute_structural_embedding(d["G"], dim=k, seed=seed)
    U = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(Z).U
    return Result(cd.defuzzify(U), _node_to_comm_view(U), k)


def fcm_content(d, seed):
    """Fuzzy c-means directly on concatenated content (no structure, no CCA
    fusion, no ANFIS confidence weighting) - the fuzzy analogue of
    kmeans_content, added as a lightweight comparable "fuzzy + multimodal"
    baseline under much weaker assumptions than NF-MCD (reviewer request:
    a comparable method that is simultaneously fuzzy/overlapping and
    multimodal, not just generic k-means/Louvain/spectral)."""
    k = d["n_communities"]
    X = content_matrix(d)
    U = cd.FuzzyCMeans(n_clusters=k, m=1.5, seed=seed).fit(X).U
    return Result(cd.defuzzify(U), _node_to_comm_view(U), k, note="fcm(content only, no structure)")


def _node_to_comm_view(U: np.ndarray) -> List[Set[int]]:
    view: List[Set[int]] = [set() for _ in range(U.shape[1])]
    for i, comms in enumerate(cd.overlapping_communities(U, OVERLAP_THRESHOLD)):
        for c in comms:
            view[c].add(i)
    return view


def _overlap_result(communities: List[List[int]], n: int, note: str) -> Result:
    view = [set(int(v) for v in c) for c in communities if len(c) > 0]
    labels = np.full(n, -1, dtype=int)
    for c, members in enumerate(view):
        for v in members:
            if labels[v] < 0:
                labels[v] = c
    next_label = len(view)
    for i in range(n):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
    return Result(labels, view, len(view), note=note)


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def demon(d, seed):
    from cdlib import algorithms

    _seed_all(seed)
    G = d["G"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nc = algorithms.demon(G, min_com_size=3, epsilon=0.25)
    return _overlap_result(nc.communities, G.number_of_nodes(), "cdlib demon(eps=0.25,min=3)")


def slpa(d, seed):
    from cdlib import algorithms

    _seed_all(seed)
    G = d["G"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nc = algorithms.slpa(G, t=21, r=0.1)
    return _overlap_result(nc.communities, G.number_of_nodes(), "cdlib slpa(t=21,r=0.1)")


# ---------------------------------------------------------------------------
# Registry: name -> (function, applies-to datasets)
# ---------------------------------------------------------------------------

ALL = {"crisismmd", "pheme", "fakeddit", "dblp", "amazon"}
TEXT = {"crisismmd", "pheme", "fakeddit"}
IMAGE = {"crisismmd", "fakeddit"}

METHODS: Dict[str, tuple] = {
    "nfmcd_full": (nfmcd_full, ALL),
    "nfmcd_scan_k": (nfmcd_scan_k, ALL),
    "nfmcd_text_only": (nfmcd_text_only, TEXT),
    "nfmcd_image_only": (nfmcd_image_only, IMAGE),
    "nfmcd_alpha_0.5": (nfmcd_alpha_fixed, TEXT),
    "nfmcd_structure_only": (nfmcd_structure_only, ALL),
    "nfmcd_linear_confidence": (nfmcd_linear_confidence, IMAGE),
    "kmeans_content": (kmeans_content, TEXT),
    "fcm_content": (fcm_content, TEXT),
    "louvain": (louvain, ALL),
    "spectral_graph+content": (spectral_graph_content, ALL),
    "fcm_structure_only": (fcm_structure_only, ALL),
    "demon": (demon, ALL),
    "slpa": (slpa, ALL),
}
