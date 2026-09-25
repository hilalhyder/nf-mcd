"""
Alternative soft-clustering steps for NF-MCD Stage 4 (optional; the default
"fcm" path in pipeline.py is untouched and does not use this module).

Motivation (established in diagnose_collapse.py): fuzzy c-means at m=1.5
degenerates to the uniform fixed point (every U.max == 1/k) when the fused
features carry weak cluster structure, while k-means on the same features
still finds partitions. Each clusterer below keeps memberships soft
(rows of U sum to 1) so overlap and explanations still work.

All hyperparameters are fixed a priori here (never tuned per dataset):
  TAU_SCALE       softmax temperature = 0.5 * median squared distance to the nearest center
  M_LADDER        fuzziness ladder for fcm_adaptive_m, largest acceptable m wins
  NONDEGENERACY   accept m if mean(U.max) >= 1/k + 0.25 * (1 - 1/k)
  KNN_SPECTRAL    12 neighbours for the spectral_soft affinity graph
  GMM             spherical covariance, n_init=3, reg_covar=1e-4
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .community_detection import FCMResult

TAU_SCALE = 0.5
M_LADDER = (1.5, 1.3, 1.2, 1.1)
NONDEGENERACY = 0.25
KNN_SPECTRAL = 12

CLUSTERERS = ("fcm", "kmeans_softmax", "gmm", "fcm_adaptive_m", "spectral_soft")


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d2 = (X ** 2).sum(1)[:, None] - 2.0 * X @ C.T + (C ** 2).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def _softmax_memberships(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d2 = _sqdist(X, C)
    tau = TAU_SCALE * max(float(np.median(d2.min(axis=1))), 1e-12)
    z = -d2 / tau
    z -= z.max(axis=1, keepdims=True)
    U = np.exp(z)
    return U / U.sum(axis=1, keepdims=True)


def _kmeans(X: np.ndarray, k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def kmeans_softmax(X: np.ndarray, k: int, seed: int) -> FCMResult:
    km = _kmeans(X, k, seed)
    U = _softmax_memberships(X, km.cluster_centers_)
    return FCMResult(U=U, centers=km.cluster_centers_, n_iter=int(km.n_iter_), objective_history=[float(km.inertia_)])


def gmm(X: np.ndarray, k: int, seed: int) -> FCMResult:
    model = GaussianMixture(n_components=k, covariance_type="spherical", n_init=3,
                            random_state=seed, reg_covar=1e-4, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    return FCMResult(U=model.predict_proba(X), centers=model.means_, n_iter=int(model.n_iter_),
                     objective_history=[float(model.lower_bound_)])


def _fcm_from_centers(X: np.ndarray, centers: np.ndarray, m: float, max_iter: int = 200, tol: float = 1e-5):
    """Same update rules as community_detection.FuzzyCMeans, but started from given centers."""
    power = 1.0 / (m - 1.0)
    prev = np.inf
    hist = []
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        d2 = np.maximum(_sqdist(X, centers), 1e-12)
        inv = d2 ** (-power)
        U = inv / inv.sum(axis=1, keepdims=True)
        Um = U ** m
        obj = float(np.sum(Um * d2))
        hist.append(obj)
        centers = (Um.T @ X) / (Um.sum(axis=0, keepdims=True).T + 1e-12)
        if abs(prev - obj) < tol:
            break
        prev = obj
    d2 = np.maximum(_sqdist(X, centers), 1e-12)
    inv = d2 ** (-power)
    U = inv / inv.sum(axis=1, keepdims=True)
    return U, centers, n_iter, hist


def fcm_adaptive_m(X: np.ndarray, k: int, seed: int) -> FCMResult:
    """FCM started from k-means centers; m = largest ladder value whose memberships are non-degenerate."""
    C0 = _kmeans(X, k, seed).cluster_centers_
    floor = 1.0 / k + NONDEGENERACY * (1.0 - 1.0 / k)
    chosen = None
    for m in M_LADDER:
        U, C, n_iter, hist = _fcm_from_centers(X, C0.copy(), m)
        chosen = (U, C, n_iter, hist, m)
        if float(U.max(axis=1).mean()) >= floor:
            break
    U, C, n_iter, hist, m = chosen
    res = FCMResult(U=U, centers=C, n_iter=n_iter, objective_history=hist)
    res.m_used = m
    return res


def spectral_soft(X: np.ndarray, k: int, seed: int) -> FCMResult:
    n = X.shape[0]
    kk = min(KNN_SPECTRAL, n - 1)
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    A = nn.kneighbors_graph(None, n_neighbors=kk, mode="connectivity")
    A = A.maximum(A.T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E = spectral_embedding(A, n_components=k, random_state=seed, drop_first=False)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return kmeans_softmax(E, k, seed)


_FUNCS = {
    "kmeans_softmax": kmeans_softmax,
    "gmm": gmm,
    "fcm_adaptive_m": fcm_adaptive_m,
    "spectral_soft": spectral_soft,
}


def cluster(X: np.ndarray, k: int, method: str, seed: int) -> FCMResult:
    if method not in _FUNCS:
        raise ValueError(f"clusterer must be one of {CLUSTERERS}, got {method!r}")
    return _FUNCS[method](X, k, seed)
