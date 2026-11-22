"""Existing K-means accelerators as step-by-step engines, for experiments/accelerators_markov.py.

Every engine follows the interface of markov_kmeans._engines: after construction and after
each ``step()`` it holds ``labels`` (the assignment of every point) and ``centers`` (the
centers those labels were computed against); ``step()`` returns the number of labels that
changed and leaves their indices (ascending) in ``movers``; ``nearest(C, inplace)`` assigns
every point to the nearest of C, pruned by the engine's own bounds, and returns
``(labels as int64, n_dist)`` (n_dist = 0 where the engine does not count distances).

    SkLloyd     scikit-learn's Lloyd kernel (chunked GEMM, OpenMP): the engine of KMeans(algorithm="lloyd")
    SkElkan     scikit-learn's Elkan kernel (one upper and k lower bounds per point): KMeans(algorithm="elkan")
    Hamerly     Hamerly (SDM 2010) with incremental center sums, as in Hamerly's reference code
    Yinyang     Yinyang K-means (Ding et al., ICML 2015): centers in G = ceil(k / 10) groups, one lower
                bound per group, global, group and local filters; incremental center sums

The scikit-learn engines call the library's own compiled iteration kernels (the same calls
as sklearn.cluster._kmeans._kmeans_single_lloyd / _elkan) so that the loop can be observed
after every iteration; a step also does what scikit-learn's loop does per iteration
(copy the labels, compare them, the center half-distances for Elkan). The two numba
engines keep running center sums, updated from the movers only, so that an iteration in
the tail costs O(n) bound tests rather than an O(n d) recomputation of the centroids.

Lloyd, Elkan, Hamerly and Yinyang produce the same sequence of partitions in exact
arithmetic; in floating point the centers differ in the last bits (summation order), and
scikit-learn moves an emptied cluster to a far point where the others keep its center, so
trajectories can part. experiments/accelerators_markov.py checks and reports this.
"""
from __future__ import annotations

import copy

import numpy as np
from numba import njit, prange

from markov_kmeans._kernels import (_SAFE, _half_nearest_center, _init, _update_bounds, center_drift,
                                    sqdist)

_CHUNKS = 256


def clone(engine):
    """Copy of an engine whose arrays (all but the data) are independent of the original."""
    c = copy.copy(engine)
    for name, v in vars(engine).items():
        if isinstance(v, np.ndarray) and v is not engine.X:
            setattr(c, name, v.copy())
    return c


# ---------------------------------------------------------------------------------------------
# scikit-learn kernels


class _SkBase:
    exact = True

    def __init__(self, X):
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        self.X = X
        self.w = np.ones(X.shape[0], dtype=X.dtype)
        self.n_threads = _openmp_effective_n_threads()
        self.n_dist = 0
        self.movers = np.empty(0, dtype=np.int64)

    def _after(self, old):
        # scikit-learn compares and copies the labels every iteration; the movers come from the same pass
        self.movers = np.flatnonzero(self.labels != old)
        return len(self.movers)


class SkLloyd(_SkBase):
    """scikit-learn's Lloyd iteration (sklearn.cluster._k_means_lloyd)."""

    def __init__(self, X, C0):
        from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense

        super().__init__(X)
        self._iter = lloyd_iter_chunked_dense
        k = C0.shape[0]
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        self.next = np.empty_like(self.centers)
        self.wic = np.zeros(k)
        self.shift = np.zeros(k)
        self.labels = np.full(X.shape[0], -1, dtype=np.int32)
        # assign against C0 and form C1 (scikit-learn's first iteration)
        self._iter(X, self.w, self.centers, self.next, self.wic, self.labels, self.shift, self.n_threads)

    def step(self):
        old = self.labels.copy()
        self.centers, self.next = self.next, self.centers           # labels are now computed against C_t
        self._iter(self.X, self.w, self.centers, self.next, self.wic, self.labels, self.shift, self.n_threads)
        return self._after(old)

    def nearest(self, C, inplace=False):
        labels = self.labels if inplace else self.labels.copy()
        C = np.ascontiguousarray(C, dtype=np.float64)
        self._iter(self.X, self.w, C, C, self.wic, labels, self.shift, self.n_threads, update_centers=False)
        return labels.astype(np.int64), self.X.shape[0] * C.shape[0]


class SkElkan(_SkBase):
    """scikit-learn's Elkan iteration (sklearn.cluster._k_means_elkan)."""

    def __init__(self, X, C0):
        from sklearn.cluster._k_means_elkan import elkan_iter_chunked_dense, init_bounds_dense

        super().__init__(X)
        self._iter = elkan_iter_chunked_dense
        n, k = X.shape[0], C0.shape[0]
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        self.next = np.empty_like(self.centers)
        self.wic = np.zeros(k)
        self.shift = np.zeros(k)
        self.labels = np.full(n, -1, dtype=np.int32)
        self.upper = np.zeros(n)
        self.lower = np.zeros((n, k))
        self._half(self.centers)
        init_bounds_dense(X, self.centers, self.chd, self.labels, self.upper, self.lower, self.n_threads)
        self._call(self.centers, self.next)                 # scikit-learn's first iteration: forms C1, moves the bounds
        self._half(self.next)

    def _half(self, C):
        from sklearn.metrics.pairwise import euclidean_distances

        self.chd = euclidean_distances(C) / 2
        self.dnc = np.partition(np.asarray(self.chd), kth=1, axis=0)[1] if C.shape[0] > 1 else np.full(1, np.inf)

    def _call(self, C_old, C_new, update_centers=True, labels=None, upper=None, lower=None):
        self._iter(self.X, self.w, C_old, C_new, self.wic, self.chd, self.dnc,
                   self.upper if upper is None else upper, self.lower if lower is None else lower,
                   self.labels if labels is None else labels, self.shift, self.n_threads,
                   update_centers=update_centers)

    def step(self):
        old = self.labels.copy()
        self.centers, self.next = self.next, self.centers
        self._call(self.centers, self.next)                 # assign against C_t, form C_{t+1}, move the bounds
        self._half(self.next)
        return self._after(old)

    def nearest(self, C, inplace=False):
        # the bounds refer to C_{t+1}; move them to C, then one assignment pass without a center update
        C = np.ascontiguousarray(C, dtype=np.float64)
        labels, upper, lower = ((self.labels, self.upper, self.lower) if inplace
                                else (self.labels.copy(), self.upper.copy(), self.lower.copy()))
        shift = np.sqrt(((C - self.next) ** 2).sum(1))
        upper += shift[labels]
        lower -= shift
        np.maximum(lower, 0.0, out=lower)
        self._half(C)
        self._call(C, C, update_centers=False, labels=labels, upper=upper, lower=lower)
        return labels.astype(np.int64), 0


# ---------------------------------------------------------------------------------------------
# numba engines with incremental center sums


@njit(cache=True)
def _sums(X, labels, k):
    n, d = X.shape
    S = np.zeros((k, d))
    cnt = np.zeros(k, dtype=np.int64)
    for i in range(n):
        c = labels[i]
        cnt[c] += 1
        for t in range(d):
            S[c, t] += X[i, t]
    return S, cnt


@njit(cache=True)
def _centers_from(S, cnt, C_old):
    k, d = S.shape
    C = np.empty((k, d))
    for j in range(k):
        if cnt[j] == 0:
            for t in range(d):
                C[j, t] = C_old[j, t]
        else:
            for t in range(d):
                C[j, t] = S[j, t] / cnt[j]
    return C


@njit(cache=True)
def _move_sums(X, movers, old, labels, S, cnt):
    d = X.shape[1]
    for q in range(movers.shape[0]):
        i = movers[q]
        a = old[q]
        b = labels[i]
        cnt[a] -= 1
        cnt[b] += 1
        for t in range(d):
            S[a, t] -= X[i, t]
            S[b, t] += X[i, t]


@njit(cache=True, parallel=True)
def _compact2(buf, obuf, cnt, n):
    off = np.zeros(_CHUNKS + 1, dtype=np.int64)
    for c in range(_CHUNKS):
        off[c + 1] = off[c] + cnt[c]
    out = np.empty(off[_CHUNKS], dtype=np.int64)
    old = np.empty(off[_CHUNKS], dtype=np.int64)
    for c in prange(_CHUNKS):
        lo = c * n // _CHUNKS
        for q in range(cnt[c]):
            out[off[c] + q] = buf[lo + q]
            old[off[c] + q] = obuf[lo + q]
    return out, old


@njit(cache=True)
def _scan(X, i, C):
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


@njit(cache=True, parallel=True)
def _hamerly_assign(X, C, labels, u, l, s, buf, obuf, cnt, nd_c):
    n = X.shape[0]
    k = C.shape[0]
    for c in prange(_CHUNKS):
        lo, hi = c * n // _CHUNKS, (c + 1) * n // _CHUNKS
        m = 0
        nd = 0
        for i in range(lo, hi):
            a = labels[i]
            mm = max(s[a], l[i])
            if u[i] * _SAFE < mm:
                continue
            u[i] = np.sqrt(sqdist(X, i, C, a))
            nd += 1
            if u[i] * _SAFE < mm:
                continue
            bj, best, second = _scan(X, i, C)
            nd += k
            u[i] = np.sqrt(best)
            l[i] = np.sqrt(second)
            if bj != a:
                labels[i] = bj
                buf[lo + m] = i
                obuf[lo + m] = a
                m += 1
        cnt[c] = m
        nd_c[c] = nd
    return nd_c.sum()


class _Incremental:
    exact = True

    def _setup(self, X, C0):
        n, k = X.shape[0], C0.shape[0]
        self.X = X
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        self.labels = np.empty(n, dtype=np.int64)
        self._buf = np.empty(n, dtype=np.int64)
        self._obuf = np.empty(n, dtype=np.int64)
        self._cnt = np.zeros(_CHUNKS, dtype=np.int64)
        self._nd = np.zeros(_CHUNKS, dtype=np.int64)
        self.movers = np.empty(0, dtype=np.int64)
        self.n_dist = n * k

    def _new_centers(self):
        C = _centers_from(self.S, self.cnt, self.centers)
        drift = center_drift(self.centers, C)
        self.centers = C
        return drift

    def _finish_step(self, nd):
        self.movers, old = _compact2(self._buf, self._obuf, self._cnt, self.X.shape[0])
        _move_sums(self.X, self.movers, old, self.labels, self.S, self.cnt)
        self.n_dist += int(nd)
        return len(self.movers)


class Hamerly(_Incremental):
    """Hamerly (SDM 2010): one upper and one lower bound per point; incremental center sums."""

    def __init__(self, X, C0):
        self._setup(X, C0)
        n = X.shape[0]
        self.u = np.empty(n)
        self.l = np.empty(n)
        _init(X, self.centers, self.labels, self.u, self.l)
        self.S, self.cnt = _sums(X, self.labels, self.centers.shape[0])

    def step(self):
        drift = self._new_centers()
        _update_bounds(self.labels, self.u, self.l, drift)
        nd = _hamerly_assign(self.X, self.centers, self.labels, self.u, self.l, _half_nearest_center(self.centers),
                             self._buf, self._obuf, self._cnt, self._nd)
        return self._finish_step(nd)

    def nearest(self, C, inplace=False):
        labels, u, l = (self.labels, self.u, self.l) if inplace else (self.labels.copy(), self.u.copy(), self.l.copy())
        C = np.ascontiguousarray(C, dtype=np.float64)
        _update_bounds(labels, u, l, center_drift(self.centers, C))
        nd = _hamerly_assign(self.X, C, labels, u, l, _half_nearest_center(C), self._buf, self._obuf,
                             self._cnt, self._nd)
        return labels, int(nd)


@njit(cache=True, parallel=True)
def _yy_init(X, C, gidx, gstart, labels, u, lb):
    n = X.shape[0]
    G = gstart.shape[0] - 1
    k = C.shape[0]
    for i in prange(n):
        dd = np.empty(k)
        bj = 0
        for j in range(k):
            dd[j] = sqdist(X, i, C, j)
            if dd[j] < dd[bj]:
                bj = j
        labels[i] = bj
        u[i] = np.sqrt(dd[bj])
        for g in range(G):
            v = np.inf
            for q in range(gstart[g], gstart[g + 1]):
                j = gidx[q]
                if j != bj and dd[j] < v:
                    v = dd[j]
            lb[i, g] = np.sqrt(v)


@njit(cache=True, parallel=True)
def _yy_assign(X, C, labels, u, lb, drift, gdrift, s, grp, gidx, gstart, buf, obuf, cnt, nd_c):
    """One Yinyang pass: move the bounds by the drifts, then global, group and local filters."""
    n = X.shape[0]
    G = gstart.shape[0] - 1
    for c in prange(_CHUNKS):
        lo, hi = c * n // _CHUNKS, (c + 1) * n // _CHUNKS
        m = 0
        nd = 0
        lbold = np.empty(G)
        m1 = np.empty(G)
        m2 = np.empty(G)
        m1i = np.empty(G, dtype=np.int64)
        scanned = np.zeros(G, dtype=np.bool_)
        for i in range(lo, hi):
            a = labels[i]
            u[i] += drift[a]
            glb = np.inf
            for g in range(G):
                lbold[g] = lb[i, g]
                lb[i, g] -= gdrift[g]
                if lb[i, g] < glb:
                    glb = lb[i, g]
            thr = max(glb, s[a])
            if u[i] * _SAFE < thr:                       # global filter
                continue
            da = np.sqrt(sqdist(X, i, C, a))
            nd += 1
            u[i] = da
            if da * _SAFE < thr:
                continue
            best = a
            bd = da
            for g in range(G):
                scanned[g] = False
                if lb[i, g] > bd * _SAFE:                # group filter
                    continue
                scanned[g] = True
                v1 = np.inf
                v2 = np.inf
                v1i = -1
                for q in range(gstart[g], gstart[g + 1]):
                    j = gidx[q]
                    if j == a:
                        val = da
                    else:
                        bnd = lbold[g] - drift[j]           # local filter
                        if bnd > bd * _SAFE:
                            val = bnd
                        else:
                            val = np.sqrt(sqdist(X, i, C, j))
                            nd += 1
                            if val < bd or (val == bd and j < best):
                                bd = val
                                best = j
                    if val < v1:
                        v2 = v1
                        v1 = val
                        v1i = j
                    elif val < v2:
                        v2 = val
                m1[g] = v1
                m2[g] = v2
                m1i[g] = v1i
            for g in range(G):
                if scanned[g]:
                    lb[i, g] = m2[g] if m1i[g] == best else m1[g]
            if best != a:
                ga = grp[a]
                if not scanned[ga] and da < lb[i, ga]:
                    lb[i, ga] = da
                labels[i] = best
                u[i] = bd
                buf[lo + m] = i
                obuf[lo + m] = a
                m += 1
        cnt[c] = m
        nd_c[c] = nd
    return nd_c.sum()


def center_groups(C0, G, seed=0):
    """Groups of the initial centers (Ding et al.: k-means on the centers, 5 iterations)."""
    k = C0.shape[0]
    if G <= 1:
        return np.zeros(k, dtype=np.int64)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=G, n_init=1, max_iter=5, random_state=seed).fit(C0).labels_.astype(np.int64)


class Yinyang(_Incremental):
    """Yinyang K-means (Ding, Zhao, Shen, Musuvathi & Mytkowicz, ICML 2015)."""

    def __init__(self, X, C0, n_groups=None):
        self._setup(X, C0)
        n, k = X.shape[0], C0.shape[0]
        G = int(n_groups or max(1, int(np.ceil(k / 10))))
        self.grp = center_groups(self.centers, G)
        G = int(self.grp.max()) + 1
        self.gidx = np.argsort(self.grp, kind="stable").astype(np.int64)
        self.gstart = np.searchsorted(self.grp[self.gidx], np.arange(G + 1)).astype(np.int64)
        self.u = np.empty(n)
        self.lb = np.empty((n, G))
        _yy_init(X, self.centers, self.gidx, self.gstart, self.labels, self.u, self.lb)
        self.S, self.cnt = _sums(X, self.labels, k)

    def _group_drift(self, drift):
        G = self.gstart.shape[0] - 1
        return np.array([drift[self.gidx[self.gstart[g]:self.gstart[g + 1]]].max() for g in range(G)])

    def _pass(self, C, labels, u, lb, drift):
        return _yy_assign(self.X, C, labels, u, lb, drift, self._group_drift(drift), _half_nearest_center(C),
                          self.grp, self.gidx, self.gstart, self._buf, self._obuf, self._cnt, self._nd)

    def step(self):
        drift = self._new_centers()
        nd = self._pass(self.centers, self.labels, self.u, self.lb, drift)
        return self._finish_step(nd)

    def nearest(self, C, inplace=False):
        labels, u, lb = (self.labels, self.u, self.lb) if inplace else (self.labels.copy(), self.u.copy(), self.lb.copy())
        C = np.ascontiguousarray(C, dtype=np.float64)
        nd = self._pass(C, labels, u, lb, center_drift(self.centers, C))
        return labels, int(nd)
