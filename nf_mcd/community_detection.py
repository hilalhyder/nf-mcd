"""
Stage 4 (Section 4.4): Soft / overlapping community detection.

The fused (content + structure) feature matrix from nf_mcd.topology feeds a
fuzzy c-means (FCM) objective adapted to graph-structured data, producing
the soft membership matrix U = [mu_ik] described in Section 3: mu_ik in
[0, 1], sum_k mu_ik = 1 for every node. Because membership degrees are
continuous, overlapping communities emerge directly by thresholding U,
without a separate post-processing step.

This module implements standard fuzzy c-means from first principles (no
external dependency such as scikit-fuzzy) so the objective and update rules
are fully transparent and easy to swap out (e.g. for a fuzzy modularity
optimizer) if you extend the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np


@dataclass
class FCMResult:
    U: np.ndarray                 # (n, k) soft membership matrix, rows sum to 1
    centers: np.ndarray           # (k, d) cluster centers in the fused feature space
    n_iter: int
    objective_history: List[float]


class FuzzyCMeans:
    """Fuzzy c-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of communities k.
    m : float, default 1.5
        Fuzziness exponent (m > 1). Larger m -> softer (more overlapping)
        memberships; m -> 1 approaches hard k-means.

        NOTE on the default: the "textbook" FCM default of m=2.0 is
        calibrated for arbitrary-scale data; on the unit-norm-ish,
        concatenated structure+content feature vectors this pipeline
        produces (nf_mcd.topology.fuse_features), m=2.0 was found
        empirically to produce memberships so soft they are numerically
        indistinguishable from uniform (max membership pinned at ~1/k for
        every node) even when the *ranking* of memberships (and therefore
        the defuzzified hard partition) is essentially correct. This is a
        genuine property of the FCM objective at this feature scale, not a
        bug: relative distances between a point and each center are small
        compared to their absolute distance from the origin, and m=2's
        squared-ratio membership formula amplifies that flatness. m=1.5
        was found to recover both a correct hard partition AND informative,
        usefully-graded soft memberships on this pipeline's feature scale
        (validated against a k-means/spectral-clustering baseline on
        synthetic data with known ground truth - see demo.py). If you
        change common_dim/structural_dim or the feature scale substantially,
        re-check this choice: inspect `U.max(axis=1)` after fitting - if it
        is pinned near 1/k for nearly every node, m is too high for your
        feature scale.
    max_iter, tol : stopping criteria on the FCM objective.
    """

    def __init__(self, n_clusters: int, m: float = 1.5, max_iter: int = 200, tol: float = 1e-5, seed: int = 0):
        assert n_clusters >= 2, "n_clusters must be >= 2"
        assert m > 1.0, "fuzziness exponent m must be > 1"
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ seeding: pick well-separated starting centers from the
        data itself.

        FCM is sensitive to initialization. Starting from a random
        *membership* matrix (the textbook-simplest approach) causes every
        initial cluster center to be a weighted average over most of the
        dataset, which - unless clusters are extremely well separated -
        lands all k initial centers very close to the single global mean.
        The subsequent iteration can then get stuck in that degenerate,
        near-uniform-membership local optimum, especially at higher
        fuzziness m (verified empirically: this collapse reproduces
        reliably on realistically-separated data, while only "escaping" it
        on synthetic clusters separated by an unrealistically large
        margin). k-means++ seeding avoids this by construction.
        """
        n = X.shape[0]
        k = self.n_clusters
        centers = np.empty((k, X.shape[1]))
        first = rng.integers(n)
        centers[0] = X[first]
        closest_dist2 = np.sum((X - centers[0]) ** 2, axis=1)

        for i in range(1, k):
            probs = closest_dist2 / (closest_dist2.sum() + 1e-12)
            next_idx = rng.choice(n, p=probs)
            centers[i] = X[next_idx]
            new_dist2 = np.sum((X - centers[i]) ** 2, axis=1)
            closest_dist2 = np.minimum(closest_dist2, new_dist2)

        return centers

    def fit(self, X: np.ndarray) -> FCMResult:
        n, d = X.shape
        k = self.n_clusters
        rng = np.random.default_rng(self.seed)

        centers = self._kmeanspp_init(X, rng)

        history: List[float] = []
        prev_obj = np.inf
        n_iter = 0
        U = None

        for it in range(self.max_iter):
            n_iter = it + 1

            # Squared distances from every point to every (current) center.
            dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (n, k)
            dist2 = np.maximum(dist2, 1e-12)

            # Update memberships from the current centers.
            power = 1.0 / (self.m - 1.0)
            inv_dist = dist2 ** (-power)                        # (n, k)
            U = inv_dist / inv_dist.sum(axis=1, keepdims=True)
            Um = U ** self.m

            # FCM objective: sum_i sum_k mu_ik^m * dist2_ik
            obj = float(np.sum(Um * dist2))
            history.append(obj)

            # Update cluster centers: weighted mean of points.
            centers = (Um.T @ X) / (Um.sum(axis=0, keepdims=True).T + 1e-12)

            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

        # Final membership at the converged centers.
        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        dist2 = np.maximum(dist2, 1e-12)
        power = 1.0 / (self.m - 1.0)
        inv_dist = dist2 ** (-power)
        U = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        return FCMResult(U=U, centers=centers, n_iter=n_iter, objective_history=history)


def fuzzy_partition_coefficient(U: np.ndarray) -> float:
    """FPC in (1/k, 1]; closer to 1 means crisper (less ambiguous) partitions.
    Useful for a rough, unsupervised scan over candidate values of k.
    """
    n = U.shape[0]
    return float(np.sum(U ** 2) / n)


def select_k_by_fpc(X: np.ndarray, k_range=range(2, 9), m: float = 2.0, seed: int = 0) -> int:
    """Pick k maximizing the fuzzy partition coefficient over a candidate range.

    A simple, fast unsupervised heuristic for choosing the number of
    communities when it is not known a priori. For rigorous model
    selection, cross-check against modularity or a validation set.
    """
    best_k, best_fpc = k_range[0], -np.inf
    for k in k_range:
        res = FuzzyCMeans(n_clusters=k, m=m, seed=seed).fit(X)
        fpc = fuzzy_partition_coefficient(res.U)
        if fpc > best_fpc:
            best_fpc, best_k = fpc, k
    return best_k


def defuzzify(U: np.ndarray) -> np.ndarray:
    """Hard partition (argmax community per node), for comparison with classical methods."""
    return np.argmax(U, axis=1)


def overlapping_communities(U: np.ndarray, threshold: float = 0.2) -> List[Set[int]]:
    """For each node, the set of communities it belongs to with membership >= threshold.

    Every node is always assigned at least its strongest community, so a
    node whose memberships all fall below `threshold` (e.g. near-uniform
    rows at high fuzziness m) is never left unassigned; `threshold` then
    only controls additional overlap.
    """
    return [
        set(np.where(row >= threshold)[0].tolist()) | {int(np.argmax(row))}
        for row in U
    ]
