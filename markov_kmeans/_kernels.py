"""Compiled K-means kernels used by Markov-K-means (Lloyd, Hamerly bounds, mini-batch).

Conventions: X is C-contiguous float64 (n, d); nearest-center ties go to the lowest
index; centers are recomputed from scratch each iteration (no incremental sums), so the
Lloyd and Hamerly engines produce bit-identical iterates.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_SAFE = 1.0 + 1e-10   # relative margin on Hamerly pruning tests (rounding in bound updates)


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


@njit(cache=True)
def _sculley_update(Xb, lab, C, counts, eta):
    d = C.shape[1]
    for r in range(Xb.shape[0]):
        c = lab[r]
        counts[c] += 1.0
        step = 1.0 / counts[c] if eta <= 0.0 else eta
        for t in range(d):
            C[c, t] += step * (Xb[r, t] - C[c, t])


@njit(cache=True)
def _soft_sums(X, labels, U, S, P, k):
    n, d = X.shape
    sums = np.zeros((k, d))
    cnt = np.zeros(k)
    in_u = np.zeros(n, dtype=np.bool_)
    for r in range(U.shape[0]):
        in_u[U[r]] = True
    for i in range(n):
        if in_u[i]:
            continue
        c = labels[i]
        cnt[c] += 1.0
        for t in range(d):
            sums[c, t] += X[i, t]
    for r in range(U.shape[0]):
        i = U[r]
        for q in range(S.shape[1]):
            j = S[r, q]
            if j < 0:
                break
            w = P[r, q]
            if w == 0.0:
                continue
            cnt[j] += w
            for t in range(d):
                sums[j, t] += w * X[i, t]
    return sums, cnt


def soft_centers(X, labels, U, S, P, C_fallback):
    """Weighted centroids: points outside U count fully for their label, U by (S, P) weights."""
    sums, cnt = _soft_sums(X, labels, U, S, P, C_fallback.shape[0])
    C = C_fallback.copy()
    ok = cnt > 0
    C[ok] = sums[ok] / cnt[ok, None]
    return C
