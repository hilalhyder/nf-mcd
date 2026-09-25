"""
Stage 3 (Section 4.3): Fuzzy integration of fused content with network topology.

Two things happen here:

  1. Nodes are given a structural embedding Z_s (via spectral embedding of
     the graph Laplacian), independent of content.
  2. A per-node fuzzy trust weight alpha_i in [alpha_min, alpha_max] is
     derived from the fusion confidence c_i (nf_mcd.fuzzy_fusion): nodes
     with high cross-modal agreement lean on their (reliable) multimodal
     content; nodes with low or missing-modality confidence lean on
     structure instead. alpha_i is never allowed to hit exactly 0 or 1, so
     structure and content are never *entirely* discarded for any node.

The paper's Section 4.3 describes a pairwise fuzzy weight alpha_ij and a
fused similarity matrix W_fused = alpha_ij * W_c + (1 - alpha_ij) * W_s.
Both a similarity-matrix version (`fuse_similarity`) and a feature-matrix
version (`fuse_features`, used by the default fuzzy c-means pipeline in
nf_mcd.community_detection) are provided here; the feature-matrix version
is the practical default because fuzzy c-means operates on features rather
than a full n x n similarity matrix, which does not scale to large graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def compute_structural_embedding(G: nx.Graph, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Spectral embedding of the (normalized) graph Laplacian.

    Nodes that are structurally similar (densely interconnected, similar
    neighborhoods) end up close together in this embedding. Falls back to
    a degree-based embedding for very small or disconnected graphs where a
    full spectral decomposition of the requested size is not available.

    IMPORTANT - choosing `dim`: classical spectral clustering (Ng-Jordan-
    Weiss) uses exactly k eigenvectors for k target clusters, not an
    arbitrarily larger number. Requesting many more eigenvectors than
    there are true communities is not "extra detail" - most of the added
    dimensions have no cluster-relevant signal, and this function's
    row-normalization step (needed so downstream Euclidean/cosine
    similarity behaves sensibly) then dilutes the real signal across those
    extra noise dimensions, degrading separability sharply (verified
    empirically: on a 4-community graph where 3-8 dimensions cleanly
    recover communities via k-means, 32 dimensions drops to near-chance
    accuracy). Prefer dim close to the expected number of communities;
    `nf_mcd.pipeline.NFMCD` defaults `structural_dim` to `n_communities`
    for exactly this reason. If you don't know the community count in
    advance, use `nf_mcd.community_detection.select_k_by_fpc` (or a
    modularity/eigengap scan) to estimate it first, rather than requesting
    a large `dim` "to be safe".
    """
    nodes = list(G.nodes())
    n = len(nodes)
    dim = max(1, min(dim, n - 2)) if n > 2 else 1

    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(float)
    try:
        # smallest-magnitude eigenvectors of the normalized Laplacian
        # (skip the trivial first eigenvector)
        v0 = np.random.default_rng(seed).normal(size=n)
        vals, vecs = eigsh(csr_matrix(L), k=dim + 1, which="SM", v0=v0)
        order = np.argsort(vals)
        vecs = vecs[:, order]
        Z_s = vecs[:, 1: dim + 1]
    except Exception:
        rng = np.random.default_rng(seed)
        deg = np.array([G.degree(v) for v in nodes], dtype=float).reshape(-1, 1)
        noise = rng.normal(scale=0.01, size=(n, dim))
        Z_s = np.repeat(deg / (deg.max() + 1e-12), dim, axis=1) + noise

    norms = np.linalg.norm(Z_s, axis=1, keepdims=True)
    Z_s = Z_s / (norms + 1e-12)
    return Z_s


def compute_alpha(confidence: np.ndarray, alpha_min: float = 0.15, alpha_max: float = 0.85) -> np.ndarray:
    """Per-node fuzzy trust weight in content vs. structure, from fusion confidence.

    alpha_i close to alpha_max  -> trust the fused multimodal content more
    alpha_i close to alpha_min  -> trust the graph structure more
    (never exactly 0 or 1, so neither signal is ever fully discarded)
    """
    c = np.clip(confidence, 0.0, 1.0)
    return alpha_min + (alpha_max - alpha_min) * c


def consistency_confidence(G: nx.Graph, nodes, fused: np.ndarray, flags) -> np.ndarray:
    """Label-free content-informativeness confidence in [0, 1].

    Assumes homophily: content is informative for community structure to the
    extent a node's content resembles its neighbours' content. Per node:
    cosine(fused content, mean of neighbours' unit-norm fused content),
    converted to a percentile rank across nodes. Nodes without content, or
    without neighbours that have content, get 0. If any node is in the CCA
    space ("both"), only "both" nodes take part (single-modality nodes hold
    raw coordinates from a different space).
    """
    from scipy.stats import rankdata

    n = fused.shape[0]
    idx = {node: i for i, node in enumerate(nodes)}
    norms = np.linalg.norm(fused, axis=1)
    has_both = any(f == "both" for f in flags)
    valid = norms > 0
    if has_both:
        valid &= np.array([f == "both" for f in flags])
    unit = np.zeros_like(fused)
    unit[valid] = fused[valid] / norms[valid][:, None]

    raw = np.full(n, np.nan)
    for node in nodes:
        i = idx[node]
        if not valid[i]:
            continue
        nb = [idx[w] for w in G.neighbors(node) if w != node and valid[idx[w]]]
        if not nb:
            continue
        m = unit[nb].mean(axis=0)
        mn = np.linalg.norm(m)
        if mn > 0:
            raw[i] = float(unit[i] @ m / mn)

    conf = np.zeros(n)
    ok = ~np.isnan(raw)
    if ok.sum() > 1:
        conf[ok] = (rankdata(raw[ok]) - 1.0) / (ok.sum() - 1.0)
    elif ok.sum() == 1:
        conf[ok] = 1.0
    return conf


def _block_scale(Z: np.ndarray) -> float:
    """1/sqrt(total variance across nodes) of a feature block (1.0 if degenerate)."""
    tv = float(np.mean(np.sum((Z - Z.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    return 1.0 / np.sqrt(tv) if tv > 1e-12 else 1.0


def raw_pca_content(e_t, e_v, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Unsupervised content features: PCA of the concatenated L2-normalised raw
    text and image embeddings (a missing modality contributes a zero block),
    projected to `dim` components and row-normalised to unit norm (like the
    structural embedding). Nodes with no content stay at the zero vector.
    Used only when NFMCD(content_features="raw_pca").
    """
    from sklearn.decomposition import PCA

    n = len(e_t)
    blocks = []
    for vecs in (e_t, e_v):
        d0 = next((v.shape[0] for v in vecs if v is not None), None)
        if d0 is None:
            continue
        M = np.zeros((n, d0))
        for i, v in enumerate(vecs):
            if v is not None:
                M[i] = v / (np.linalg.norm(v) + 1e-12)
        blocks.append(M)
    if not blocks:
        return np.zeros((n, 1))
    X = np.hstack(blocks)
    has = np.linalg.norm(X, axis=1) > 0
    d = max(1, min(dim, int(has.sum()) - 1, X.shape[1]))
    pca = PCA(n_components=d, random_state=seed).fit(X[has])
    Z = np.zeros((n, d))
    Z[has] = pca.transform(X[has])
    nz = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (nz + 1e-12)


def fuse_features(Z_c: np.ndarray, Z_s: np.ndarray, alpha: np.ndarray, standardize_blocks: bool = False) -> np.ndarray:
    """Node-level feature fusion used by the default (fuzzy c-means) pipeline.

    Each node's content and structural embeddings are scaled by sqrt(alpha_i)
    and sqrt(1 - alpha_i) respectively before concatenation, so that
    Euclidean distance in the fused space approximately reflects the
    alpha-weighted combination of content and structural (dis)similarity.

    With standardize_blocks=True each block is first rescaled to unit total
    variance across nodes, so neither block dominates just because of scale.
    """
    n = Z_c.shape[0]
    assert Z_s.shape[0] == n and alpha.shape[0] == n
    if standardize_blocks:
        Z_c = Z_c * _block_scale(Z_c)
        Z_s = Z_s * _block_scale(Z_s)
    a = alpha.reshape(-1, 1)
    fused = np.concatenate([np.sqrt(a) * Z_c, np.sqrt(1.0 - a) * Z_s], axis=1)
    return fused


def fuse_similarity(W_c: np.ndarray, W_s: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pairwise similarity fusion: W_fused = alpha_ij * W_c + (1 - alpha_ij) * W_s,
    with alpha_ij = (alpha_i + alpha_j) / 2, exactly as formalized in Section 4.3.

    Provided for similarity-matrix-based community detection (e.g. fuzzy
    spectral clustering) on graphs small enough for an n x n matrix.
    Not used by the default pipeline (see module docstring).
    """
    a = alpha.reshape(-1, 1)
    alpha_ij = (a + a.T) / 2.0
    return alpha_ij * W_c + (1.0 - alpha_ij) * W_s


def cosine_similarity_matrix(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / (norms + 1e-12)
    return Zn @ Zn.T
