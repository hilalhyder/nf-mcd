"""
Evaluation metrics matching Section 5.3 of the paper:
  - Modularity of the defuzzified (hard) partition, for comparability with
    classical methods.
  - Overlapping Normalized Mutual Information (ONMI) against ground-truth
    or proxy community labels.
  - F1 score for community membership prediction.

`overlapping_nmi` calls cdlib's reference implementation of the LFK (2009)
overlapping NMI (requires `cdlib`). The earlier Hungarian-matched
approximation is kept as `overlapping_nmi_approx` for comparison only.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def modularity_score(G: nx.Graph, hard_labels: np.ndarray) -> float:
    """Modularity of the defuzzified hard partition (argmax community per node)."""
    nodes = list(G.nodes())
    communities: Dict[int, Set] = {}
    for node, label in zip(nodes, hard_labels):
        communities.setdefault(int(label), set()).add(node)
    partition = list(communities.values())
    return nx.algorithms.community.quality.modularity(G, partition)


def _binary_nmi(a: np.ndarray, b: np.ndarray) -> float:
    """NMI between two binary membership vectors (treated as 2-outcome variables)."""
    n = len(a)
    p1a, p1b = a.mean(), b.mean()
    if p1a in (0.0, 1.0) or p1b in (0.0, 1.0):
        return 1.0 if np.array_equal(a, b) else 0.0

    def h(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    Ha, Hb = h(p1a), h(p1b)

    p11 = np.mean((a == 1) & (b == 1))
    p10 = np.mean((a == 1) & (b == 0))
    p01 = np.mean((a == 0) & (b == 1))
    p00 = np.mean((a == 0) & (b == 0))

    def term(pxy, px, py):
        if pxy <= 0:
            return 0.0
        return pxy * np.log2(pxy / (px * py))

    mi = (
        term(p11, p1a, p1b)
        + term(p10, p1a, 1 - p1b)
        + term(p01, 1 - p1a, p1b)
        + term(p00, 1 - p1a, 1 - p1b)
    )
    denom = max((Ha + Hb) / 2.0, 1e-12)
    return float(np.clip(mi / denom, 0.0, 1.0))


def overlapping_nmi(
    pred_communities: List[Set[int]],
    true_communities: List[Set[int]],
    n_nodes: int,
) -> float:
    """Overlapping NMI (Lancichinetti-Fortunato-Kertesz 2009), computed by
    cdlib's reference implementation.

    `pred_communities` / `true_communities`: list of node-index sets, one
    set per community (see `nf_mcd.pipeline.NFMCD.overlapping_communities_view`).
    Empty communities are dropped. `n_nodes` is unused here; kept so the
    signature matches `overlapping_nmi_approx`.
    """
    from cdlib import NodeClustering, evaluation as cdlib_eval

    pred = [sorted(c) for c in pred_communities if c]
    true = [sorted(c) for c in true_communities if c]
    if not pred or not true:
        return 0.0
    pred_nc = NodeClustering(pred, graph=None, method_name="pred")
    true_nc = NodeClustering(true, graph=None, method_name="truth")
    return float(cdlib_eval.overlapping_normalized_mutual_information_LFK(pred_nc, true_nc).score)


def overlapping_nmi_approx(
    pred_communities: List[Set[int]],
    true_communities: List[Set[int]],
    n_nodes: int,
) -> float:
    """Simplified Hungarian-matched binary-NMI approximation (the project's
    original metric). Kept only for comparison in validate_onmi.py; it
    disagreed with the LFK reference by up to 0.11 and reported 0.08 vs
    0.000 on PHEME, so don't report it.
    """
    if not pred_communities or not true_communities:
        return 0.0

    def to_binary(comms):
        M = np.zeros((len(comms), n_nodes), dtype=int)
        for i, c in enumerate(comms):
            for node in c:
                M[i, node] = 1
        return M

    P = to_binary(pred_communities)
    Tt = to_binary(true_communities)

    cost = np.zeros((len(pred_communities), len(true_communities)))
    for i in range(len(pred_communities)):
        for j in range(len(true_communities)):
            cost[i, j] = -_binary_nmi(P[i], Tt[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    scores = [-cost[r, c] for r, c in zip(row_ind, col_ind)]
    return float(np.mean(scores)) if scores else 0.0


def membership_f1(
    pred_communities: List[Set[int]],
    true_communities: List[Set[int]],
) -> float:
    """Best-match F1: match each predicted community to its best-overlapping
    ground-truth community (Hungarian algorithm on pairwise F1), then
    average the matched F1 scores.
    """
    if not pred_communities or not true_communities:
        return 0.0

    n_p, n_t = len(pred_communities), len(true_communities)
    f1 = np.zeros((n_p, n_t))
    for i, p in enumerate(pred_communities):
        for j, t in enumerate(true_communities):
            if not p and not t:
                f1[i, j] = 1.0
                continue
            inter = len(p & t)
            prec = inter / len(p) if p else 0.0
            rec = inter / len(t) if t else 0.0
            f1[i, j] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    row_ind, col_ind = linear_sum_assignment(-f1)
    return float(np.mean([f1[r, c] for r, c in zip(row_ind, col_ind)]))
