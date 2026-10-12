"""Markov-chain assignment of non-converging points (the method under study).

For each unstable point the labels it took over the last W iterations are treated as
one trajectory of a Markov chain on the clusters it visited.  The transition matrix is
the (optionally Dirichlet-smoothed) count matrix of observed moves, and the point is
assigned by the chain's long-run behaviour:

* hard: the cluster with the largest long-run probability,
* soft: the long-run distribution itself, read as fractional membership.

"Long-run" is the Cesàro limit of the chain started from the point's current label,
``lim (1/n) sum_t e_last P^t``.  For an irreducible chain this is the unique stationary
distribution; for a chain whose last state was absorbing (A A A B B B) it puts all mass
on that state; for a periodic chain (A B A B) it is the time-average (0.5, 0.5).  It is
computed as the limit of the lazy chain (I + P)/2, which has the same Cesàro limit but
is aperiodic, by repeated squaring until the powers stop changing. Chains on one or two
states (most unstable points) use the closed form instead: two communicating states
a <-> b with move probabilities p (a to b) and q (b to a) have the long-run law
(q, p) / (p + q); a state with no way out absorbs everything.

A state that occurs only as the final label has no observed outgoing move; it is given
a self-loop (the point is assumed to stay where it was last seen).

The chains of different points are independent and are computed in parallel.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


_CHUNKS = 64


@njit(cache=True)
def _limit_from(P, start, n_square):
    """Row ``start`` of the limit of the lazy chain (I + P)/2, squaring until it stops changing."""
    s = P.shape[0]
    Q = 0.5 * P
    for a in range(s):
        Q[a, a] += 0.5
    for _ in range(n_square):
        Q2 = Q @ Q
        diff = 0.0
        for a in range(s):  # renormalise: rounding in row sums would otherwise double per squaring
            tot = 0.0
            for b in range(s):
                tot += Q2[a, b]
            for b in range(s):
                Q2[a, b] /= tot
                d = abs(Q2[a, b] - Q[a, b])
                if d > diff:
                    diff = d
        Q = Q2
        if diff <= 1e-15:
            break
    return Q[start].copy()


@njit(cache=True)
def _one_chain(seq, alpha, prior_rows, n_square, states, counts, idx, pi):
    """Fill ``states`` (visited labels, -1 padded) and ``pi`` (long-run probs); return #states."""
    w = seq.shape[0]
    s = 0
    for t in range(w):
        found = -1
        for q in range(s):
            if states[q] == seq[t]:
                found = q
                break
        if found < 0:
            states[s] = seq[t]
            found = s
            s += 1
        idx[t] = found
    for q in range(s, states.shape[0]):
        states[q] = -1
    if s == 1:
        pi[0] = 1.0
        return 1
    for a in range(s):
        for b in range(s):
            counts[a, b] = alpha
    for t in range(w - 1):
        counts[idx[t], idx[t + 1]] += 1.0
    if prior_rows.shape[0] > 0:
        for a in range(s):
            for b in range(s):
                counts[a, b] += prior_rows[states[a], states[b]]
    if s == 2:
        r0 = counts[0, 0] + counts[0, 1]
        r1 = counts[1, 0] + counts[1, 1]
        p = counts[0, 1] / r0 if r0 > 0.0 else 0.0     # a state without observed moves stays put
        q = counts[1, 0] / r1 if r1 > 0.0 else 0.0
        if p + q > 0.0:
            pi[0] = q / (p + q)
            pi[1] = p / (p + q)
        else:
            pi[0] = 1.0 if idx[w - 1] == 0 else 0.0
            pi[1] = 1.0 - pi[0]
        return 2
    P = np.zeros((s, s))
    for a in range(s):
        tot = 0.0
        for b in range(s):
            tot += counts[a, b]
        if tot == 0.0:
            P[a, a] = 1.0
        else:
            for b in range(s):
                P[a, b] = counts[a, b] / tot
    lim = _limit_from(P, idx[w - 1], n_square)
    for q in range(s):
        pi[q] = lim[q]
    return s


@njit(cache=True, parallel=True)
def _chains(Hw, alpha, prior, n_square):
    w, m = Hw.shape
    S = np.full((m, w), -1, dtype=np.int64)
    Pi = np.zeros((m, w))
    for c in prange(_CHUNKS):
        counts = np.empty((w, w))
        states = np.empty(w, dtype=np.int64)
        idx = np.empty(w, dtype=np.int64)
        pi = np.empty(w)
        seq = np.empty(w, dtype=np.int64)
        for i in range(c * m // _CHUNKS, (c + 1) * m // _CHUNKS):
            for t in range(w):
                seq[t] = Hw[t, i]
            s = _one_chain(seq, alpha, prior, n_square, states, counts, idx, pi)
            for q in range(s):
                S[i, q] = states[q]
                Pi[i, q] = pi[q]
    return S, Pi


def stationary(Hw, alpha=0.0, prior=None, n_square=40):
    """Long-run distribution of each column's label trajectory.

    Hw : (W+1, m) integer array — labels of m points over the window, oldest first.
    Returns (states, probs), both (m, W+1): visited labels (-1 padded) and their
    long-run probabilities.
    """
    Hw = np.ascontiguousarray(Hw, dtype=np.int64)
    pr = np.zeros((0, 0)) if prior is None else np.ascontiguousarray(prior, dtype=np.float64)
    return _chains(Hw, float(alpha), pr, int(n_square))


def pooled_prior(Hw, k, strength=2.0):
    """Empirical-Bayes prior: pooled transition frequencies between cluster pairs.

    Row a is the distribution of observed moves out of cluster a over all points'
    windows, scaled to ``strength`` pseudo-transitions; it is added to each point's
    counts restricted to the states that point visited.
    """
    a = np.asarray(Hw[:-1], dtype=np.int64).ravel()
    b = np.asarray(Hw[1:], dtype=np.int64).ravel()
    G = np.zeros((k, k))
    np.add.at(G, (a, b), 1.0)
    tot = G.sum(1, keepdims=True)
    return np.divide(G, tot, out=np.zeros_like(G), where=tot > 0) * strength


@njit(cache=True, parallel=True)
def _frequency(Hw):
    w, m = Hw.shape
    S = np.full((m, w), -1, dtype=np.int64)
    F = np.zeros((m, w))
    for i in prange(m):
        s = 0
        for t in range(w):
            v = Hw[t, i]
            found = -1
            for q in range(s):
                if S[i, q] == v:
                    found = q
                    break
            if found < 0:
                S[i, s] = v
                F[i, s] = 1.0
                s += 1
            else:
                F[i, found] += 1.0
        for q in range(s):
            F[i, q] /= w
    return S, F


def frequency(Hw):
    """Empirical occupancy of each visited label over the window (the non-Markov baseline).

    Same (states, probs) layout as :func:`stationary`, states in order of first visit.
    """
    return _frequency(np.ascontiguousarray(Hw, dtype=np.int64))


def argmax_label(states, probs, current, tol=1e-12):
    """Hard label from (states, probs); ties go to the current label, then the lowest index."""
    valid = states >= 0
    p = np.where(valid, probs, -np.inf)
    cand = valid & (p >= p.max(1, keepdims=True) - tol)
    cur_ok = (cand & (states == current[:, None])).any(1)
    lowest = np.where(cand, states, np.iinfo(np.int64).max).min(1)
    return np.where(cur_ok, current, lowest)


def to_dense(states, probs, k):
    """(m, k) membership matrix from the sparse (states, probs) representation."""
    m = states.shape[0]
    D = np.zeros((m, k))
    rows = np.repeat(np.arange(m), states.shape[1])
    s = states.ravel()
    ok = s >= 0
    np.add.at(D, (rows[ok], s[ok]), probs.ravel()[ok])
    return D
