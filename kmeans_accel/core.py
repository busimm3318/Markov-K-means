"""Shared contract, result types and numba kernels for every K-means variant.

Contract for an *exact* (Lloyd-equivalent) algorithm ``fit(X, init_centers, max_iter)``:

    labels  = assign every point to its nearest center         (iteration 0)
    for it in 1..max_iter:
        C_new  = compute_centers(X, labels, C)                 (empty cluster keeps old center)
        labels_new = assign every point to nearest of C_new
        C = C_new
        if labels_new == labels: converged, stop
        labels = labels_new

* ``n_iter`` is the number of center updates performed.
* Nearest-center ties break toward the lowest center index (strict ``<``).
* Centers are always recomputed from scratch with :func:`compute_centers` (no
  incremental sums) so every exact method sees bit-identical centers.
* ``n_dist`` counts point-to-center distance evaluations (tree methods also count
  node-to-center evaluations here and say so in ``extra``); ``n_aux_dist`` counts
  center-to-center / center-drift evaluations.

An exact method must return the same ``labels``, ``centers`` and ``n_iter`` as
:func:`kmeans_accel.lloyd.lloyd` for the same ``init_centers``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit


@dataclass
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    n_dist: int
    n_aux_dist: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class SeedResult:
    centers: np.ndarray
    indices: np.ndarray
    n_dist: int
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def as_data(X) -> np.ndarray:
    """Return ``X`` as a C-contiguous float64 array (the only layout kernels accept)."""
    return np.ascontiguousarray(X, dtype=np.float64)


@njit(cache=True, inline="always")
def sqdist(X, i, C, j):
    s = 0.0
    for t in range(X.shape[1]):
        diff = X[i, t] - C[j, t]
        s += diff * diff
    return s


@njit(cache=True, inline="always")
def dist(X, i, C, j):
    return np.sqrt(sqdist(X, i, C, j))


@njit(cache=True)
def compute_centers(X, labels, old_centers):
    """Centroids of ``labels``; an empty cluster keeps its previous center.

    Summation order is fixed (point index order) so all methods get identical bits.
    """
    n, d = X.shape
    k = old_centers.shape[0]
    sums = np.zeros((k, d))
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
        c = labels[i]
        counts[c] += 1
        for t in range(d):
            sums[c, t] += X[i, t]
    centers = np.empty((k, d))
    for j in range(k):
        if counts[j] == 0:
            for t in range(d):
                centers[j, t] = old_centers[j, t]
        else:
            for t in range(d):
                centers[j, t] = sums[j, t] / counts[j]
    return centers, counts


@njit(cache=True)
def center_drift(old_centers, new_centers):
    """Euclidean distance each center moved (k auxiliary distance evaluations)."""
    k = old_centers.shape[0]
    out = np.empty(k)
    for j in range(k):
        out[j] = dist(old_centers, j, new_centers, j)
    return out


@njit(cache=True)
def center_pairwise(C):
    """Symmetric (k, k) matrix of center-center distances (k(k-1)/2 evaluations)."""
    k = C.shape[0]
    D = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            v = dist(C, a, C, b)
            D[a, b] = v
            D[b, a] = v
    return D


@njit(cache=True)
def inertia(X, C, labels):
    s = 0.0
    for i in range(X.shape[0]):
        s += sqdist(X, i, C, labels[i])
    return s


@njit(cache=True)
def nearest_all(X, C, labels):
    """Brute-force nearest center for every point; returns #labels changed.

    Pass ``labels`` filled with -1 to count every point as changed.
    """
    n = X.shape[0]
    k = C.shape[0]
    changed = 0
    for i in range(n):
        best = np.inf
        bj = 0
        for j in range(k):
            s = sqdist(X, i, C, j)
            if s < best:
                best = s
                bj = j
        if bj != labels[i]:
            changed += 1
            labels[i] = bj
    return changed


def check_inputs(X, init_centers):
    X = as_data(X)
    C = np.array(init_centers, dtype=np.float64, order="C", copy=True)
    if X.ndim != 2 or C.ndim != 2 or X.shape[1] != C.shape[1]:
        raise ValueError(f"shape mismatch: X {X.shape}, centers {C.shape}")
    return X, C
