"""Hamerly's exact accelerated K-means (Hamerly, "Making k-means even faster", SDM 2010).

Per point it keeps an upper bound ``u`` on the distance to its assigned center and one
lower bound ``l`` on the distance to the second-closest center; ``s[j]`` is half the
distance from center j to its nearest other center.  A point is skipped while
``u < max(s[a], l)``.  Memory is O(n), so it scales to very large n where Elkan's
O(nk) bounds do not fit.

Pruning tests are strict and carry a tiny relative safety margin so that rounding in
the bound updates can never hide a change: results are identical to
:func:`kmeans_accel.lloyd.lloyd` (lowest-index ties included).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from .core import KMeansResult, center_drift, check_inputs, compute_centers, inertia, sqdist

ALGORITHMS = {
    "hamerly": {"kind": "exact", "fit": "hamerly",
                "ref": "G. Hamerly, Making k-means even faster, SIAM SDM 2010"},
}

_SAFE = 1.0 + 1e-10


@njit(cache=True)
def _full_scan(X, i, C):
    """Nearest and second-nearest center of point i by squared distance (lowest-index ties)."""
    best = np.inf
    second = np.inf
    bj = 0
    for j in range(C.shape[0]):
        s = sqdist(X, i, C, j)
        if s < best:
            second = best
            best = s
            bj = j
        elif s < second:
            second = s
    return bj, best, second


@njit(cache=True)
def _init(X, C, labels, u, l):
    for i in range(X.shape[0]):
        bj, best, second = _full_scan(X, i, C)
        labels[i] = bj
        u[i] = np.sqrt(best)
        l[i] = np.sqrt(second)


@njit(cache=True)
def _half_nearest_center(C):
    k = C.shape[0]
    s = np.full(k, np.inf)
    for a in range(k):
        for b in range(a + 1, k):
            v = 0.5 * np.sqrt(sqdist(C, a, C, b))
            if v < s[a]:
                s[a] = v
            if v < s[b]:
                s[b] = v
    return s


@njit(cache=True)
def _update_bounds(labels, u, l, drift):
    k = drift.shape[0]
    m1 = -1.0
    m2 = -1.0
    j1 = -1
    for j in range(k):
        if drift[j] > m1:
            m2 = m1
            m1 = drift[j]
            j1 = j
        elif drift[j] > m2:
            m2 = drift[j]
    if k == 1:
        m2 = 0.0
    for i in range(labels.shape[0]):
        a = labels[i]
        u[i] += drift[a]
        l[i] -= m2 if a == j1 else m1


@njit(cache=True)
def _assign(X, C, labels, u, l, s):
    """One Hamerly assignment pass; returns (#labels changed, #point-center distances)."""
    k = C.shape[0]
    changed = 0
    n_dist = 0
    for i in range(X.shape[0]):
        a = labels[i]
        m = max(s[a], l[i])
        if u[i] * _SAFE < m:
            continue
        u[i] = np.sqrt(sqdist(X, i, C, a))
        n_dist += 1
        if u[i] * _SAFE < m:
            continue
        bj, best, second = _full_scan(X, i, C)
        n_dist += k
        u[i] = np.sqrt(best)
        l[i] = np.sqrt(second)
        if bj != a:
            labels[i] = bj
            changed += 1
    return changed, n_dist


def hamerly(X, init_centers, max_iter=300, callback=None):
    """Exact Lloyd-equivalent K-means with Hamerly bounds.

    ``callback(t, labels, centers, n_dist_t, seconds_t)`` (optional) is called after the
    initial assignment (t=0) and after every iteration t with that iteration's cost.
    """
    t0 = time.perf_counter()
    X, C = check_inputs(X, init_centers)
    n, k = X.shape[0], C.shape[0]
    labels = np.empty(n, dtype=np.int64)
    u = np.empty(n)
    l = np.empty(n)
    _init(X, C, labels, u, l)
    n_dist = n * k
    n_aux = 0
    if callback is not None:
        callback(0, labels, C, n * k, time.perf_counter() - t0)
    n_iter = 0
    converged = False
    for _ in range(max_iter):
        ti = time.perf_counter()
        new_C, _ = compute_centers(X, labels, C)
        drift = center_drift(C, new_C)
        C = new_C
        s = _half_nearest_center(C)
        n_aux += k + k * (k - 1) // 2
        _update_bounds(labels, u, l, drift)
        changed, nd = _assign(X, C, labels, u, l, s)
        n_dist += nd
        n_iter += 1
        if callback is not None:
            callback(n_iter, labels, C, nd, time.perf_counter() - ti)
        if changed == 0:
            converged = True
            break
    return KMeansResult(C, labels, inertia(X, C, labels), n_iter, converged, n_dist, n_aux,
                        seconds=time.perf_counter() - t0)
