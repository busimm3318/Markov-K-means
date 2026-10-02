"""Strategies for finishing K-means once only a few points are still changing.

Every strategy starts from the identical state at iteration T0 — labels ``L`` (nearest
to the centers ``C_T`` that produced them), the label history window ``Hw`` of the
unstable set ``U`` — and returns a :class:`TailResult`.  Costs count only the work done
*after* T0: point-center distance evaluations and full O(n d) center passes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import markov_assign as mk
from .core import compute_centers, sqdist


@dataclass
class TailResult:
    labels: np.ndarray
    centers: np.ndarray
    n_dist: int = 0
    center_passes: int = 0
    seconds: float = 0.0
    soft: tuple | None = None               # sparse (states, probs), each (|U|, W+1)
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- kernels

@njit(cache=True)
def _nearest_subset(X, idx, C, out):
    for r in range(idx.shape[0]):
        i = idx[r]
        best = np.inf
        bj = 0
        for j in range(C.shape[0]):
            s = sqdist(X, i, C, j)
            if s < best:
                best = s
                bj = j
        out[r] = bj


@njit(cache=True)
def _sqdist_subset(X, idx, C):
    D = np.empty((idx.shape[0], C.shape[0]))
    for r in range(idx.shape[0]):
        for j in range(C.shape[0]):
            D[r, j] = sqdist(X, idx[r], C, j)
    return D


@njit(cache=True)
def _sums(X, labels, k):
    d = X.shape[1]
    S = np.zeros((k, d))
    cnt = np.zeros(k)
    for i in range(X.shape[0]):
        c = labels[i]
        cnt[c] += 1.0
        for t in range(d):
            S[c, t] += X[i, t]
    return S, cnt


@njit(cache=True)
def _active_set(X, idx, labels, S, cnt, C, max_iter):
    """Lloyd restricted to points idx; others frozen. Centers kept from running sums."""
    k, d = C.shape
    n_dist = 0
    it = 0
    new = np.empty(idx.shape[0], dtype=np.int64)
    for it in range(1, max_iter + 1):
        for j in range(k):
            if cnt[j] > 0:
                for t in range(d):
                    C[j, t] = S[j, t] / cnt[j]
        _nearest_subset(X, idx, C, new)
        n_dist += idx.shape[0] * k
        changed = 0
        for r in range(idx.shape[0]):
            i = idx[r]
            a, b = labels[i], new[r]
            if a != b:
                changed += 1
                labels[i] = b
                cnt[a] -= 1.0
                cnt[b] += 1.0
                for t in range(d):
                    S[a, t] -= X[i, t]
                    S[b, t] += X[i, t]
        if changed == 0:
            break
    return n_dist, it


@njit(cache=True)
def _hartigan(X, idx, labels, S, cnt, max_sweeps):
    """Hartigan single-point moves over points idx with exact centroid bookkeeping."""
    k, d = S.shape
    C = np.empty((k, d))
    for j in range(k):
        for t in range(d):
            C[j, t] = S[j, t] / cnt[j] if cnt[j] > 0 else 0.0
    n_dist = 0
    moves = 0
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        moved = 0
        for r in range(idx.shape[0]):
            i = idx[r]
            a = labels[i]
            if cnt[a] <= 1.0:
                continue
            da = sqdist(X, i, C, a)
            cost_out = cnt[a] / (cnt[a] - 1.0) * da
            best = cost_out
            bj = a
            for j in range(k):
                if j == a:
                    continue
                v = cnt[j] / (cnt[j] + 1.0) * sqdist(X, i, C, j)
                if v < best:
                    best = v
                    bj = j
            n_dist += k
            if bj != a:
                moved += 1
                for t in range(d):
                    S[a, t] -= X[i, t]
                    S[bj, t] += X[i, t]
                cnt[a] -= 1.0
                cnt[bj] += 1.0
                for t in range(d):
                    C[a, t] = S[a, t] / cnt[a]
                    C[bj, t] = S[bj, t] / cnt[bj]
                labels[i] = bj
        moves += moved
        if moved == 0:
            break
    return n_dist, moves, sweep


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


# ----------------------------------------------------------------------------- strategies

def _centroids(X, labels, C_prev):
    return compute_centers(X, labels, C_prev)[0]


def _finish(X, lab, C_T, fixed):
    """Output centers: the fixed representatives, or one centroid pass over ``lab``."""
    return (C_T.copy(), 0) if fixed else (_centroids(X, lab, C_T), 1)


def stop(X, L, C_T, U, Hw, fixed=False):
    """B1: stop at T0 and keep the current labels."""
    t = time.perf_counter()
    C, passes = _finish(X, L, C_T, fixed)
    return TailResult(L.copy(), C, 0, passes, time.perf_counter() - t)


def reassign_u(X, L, C_T, U, Hw, fixed=False):
    """B2: reassign only U to its nearest center (after one more center update unless fixed)."""
    t = time.perf_counter()
    C, passes = _finish(X, L, C_T, fixed)
    lab = L.copy()
    new = np.empty(len(U), dtype=np.int64)
    _nearest_subset(X, U, C, new)
    lab[U] = new
    if not fixed:
        C = _centroids(X, lab, C)
        passes += 1
    return TailResult(lab, C, len(U) * C.shape[0], passes, time.perf_counter() - t)


def majority(X, L, C_T, U, Hw, fixed=False, soft=False):
    """B3: most frequent label over the window (ties -> current label); soft = occupancy."""
    t = time.perf_counter()
    S, F = mk.frequency(Hw)
    lab = L.copy()
    lab[U] = mk.argmax_label(S, F, L[U])
    if soft and not fixed:
        C, passes = soft_centers(X, L, U, S, F, C_T), 1
    else:
        C, passes = _finish(X, lab, C_T, fixed)
    return TailResult(lab, C, 0, passes, time.perf_counter() - t, soft=(S, F) if soft else None)


def active_set(X, L, C_T, U, Hw, max_iter=200):
    """B4: freeze stable points, run Lloyd on U alone until U stops changing."""
    t = time.perf_counter()
    lab = L.copy()
    S, cnt = _sums(X, lab, C_T.shape[0])
    C = C_T.copy()
    nd, it = _active_set(X, U, lab, S, cnt, C, max_iter)
    C = _centroids(X, lab, C)
    return TailResult(lab, C, nd, 2, time.perf_counter() - t, extra={"iters": int(it)})


def hartigan_u(X, L, C_T, U, Hw, max_sweeps=200):
    """B5: Hartigan's single-point moves restricted to U."""
    t = time.perf_counter()
    lab = L.copy()
    S, cnt = _sums(X, lab, C_T.shape[0])
    nd, moves, sweeps = _hartigan(X, U, lab, S, cnt, max_sweeps)
    C = _centroids(X, lab, C_T)
    return TailResult(lab, C, nd, 2, time.perf_counter() - t, extra={"moves": int(moves), "sweeps": int(sweeps)})


def markov(X, L, C_T, U, Hw, alpha=0.0, window=None, pooled=0.0, soft=False, fixed=False):
    """P1/P2: per-point Markov chain over the label window; hard argmax or soft membership."""
    t = time.perf_counter()
    k = C_T.shape[0]
    H = Hw if window is None else Hw[-(window + 1):]
    prior = mk.pooled_prior(H, k, pooled) if pooled > 0 else None
    S, P = mk.stationary(H, alpha=alpha, prior=prior)
    lab = L.copy()
    lab[U] = mk.argmax_label(S, P, L[U])
    t_markov = time.perf_counter() - t
    if soft and not fixed:
        C, passes = soft_centers(X, L, U, S, P, C_T), 1
    else:
        C, passes = _finish(X, lab, C_T, fixed)
    return TailResult(lab, C, 0, passes, time.perf_counter() - t, soft=(S, P) if soft else None,
                      extra={"markov_seconds": t_markov, "mean_states": float((S >= 0).sum(1).mean())})


def reassign_all(X, L, C_T, U, Hw, chunk=1 << 16):
    """Nearest-center assignment of every point to the fixed centers C_T (n k distances)."""
    t = time.perf_counter()
    n = X.shape[0]
    lab = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk), dtype=np.int64)
        _nearest_subset(X, idx, C_T, lab[s:s + len(idx)])
    return TailResult(lab, C_T.copy(), n * C_T.shape[0], 0, time.perf_counter() - t)


# ----------------------------------------------------------------------------- soft comparators

def _fcm(D, m=2.0):
    inv = np.maximum(D, 1e-300) ** (-1.0 / (m - 1.0))
    return inv / inv.sum(1, keepdims=True)


def _gmm(D, var, w):
    logp = np.log(np.maximum(w, 1e-300)) - D / (2.0 * var)
    logp -= logp.max(1, keepdims=True)
    P = np.exp(logp)
    return P / P.sum(1, keepdims=True)


def soft_comparator(kind, X, C, U, eval_rows, params=None, chunk=100_000):
    """Distance-based memberships of U (fuzzy c-means m=2, or plug-in GMM posterior).

    Processed in chunks so |U| k never materialises; returns the argmax label of every
    point of U and the dense memberships of the rows ``eval_rows`` only.
    """
    k = C.shape[0]
    labels = np.empty(len(U), dtype=np.int64)
    W_eval = np.empty((len(eval_rows), k))
    pos = np.full(len(U), -1, dtype=np.int64)
    pos[eval_rows] = np.arange(len(eval_rows))
    for s0 in range(0, len(U), chunk):
        sl = slice(s0, min(len(U), s0 + chunk))
        D = _sqdist_subset(X, U[sl], C)
        Wm = _fcm(D) if kind == "fcm" else _gmm(D, *params)
        labels[sl] = Wm.argmax(1)
        p = pos[sl]
        W_eval[p[p >= 0]] = Wm[p >= 0]
    return labels, W_eval


def gmm_params(X, labels, C):
    """Shared isotropic variance and mixture weights implied by a hard partition."""
    from .core import inertia
    n, d = X.shape
    w = np.bincount(labels, minlength=C.shape[0]).astype(float) / n
    return inertia(X, C, labels) / (n * d), w


def soft_gmm(X, labels, C, U, params=None):
    """Plug-in GMM posterior of U: shared isotropic variance, weights from the partition."""
    var, w = gmm_params(X, labels, C) if params is None else params
    return _gmm(_sqdist_subset(X, U, C), var, w)


def true_posterior(X, U, means, var=1.0):
    """Generative posterior p(z=j | x) of the equal-weight isotropic GMM used to make X."""
    D = _sqdist_subset(X, U, np.ascontiguousarray(means, dtype=np.float64))
    logp = -D / (2.0 * var)
    logp -= logp.max(1, keepdims=True)
    P = np.exp(logp)
    return P / P.sum(1, keepdims=True)
