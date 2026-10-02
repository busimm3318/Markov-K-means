"""Decision rules of Markov-K-means: when the Markov chain intervenes, and how it settles points.

The fit loop feeds every iteration to a :class:`StateMonitor`, which judges the state of
the whole K-means run from three groups of signals:

Movement — how many vectors are still moving
    ``unstable_frac``      share of vectors whose label changed within the last W iterations
    ``oscillating_share``  share of those that switched at least twice (oscillation, not drift)
    ``projected_changes``  label changes still to come, extrapolated from the decay of the
                           per-iteration change counts, as a share of n

Representatives — how stable the centers are
    ``center_drift``       max over centers of (distance moved in the last iteration) divided by
                           half the distance to the nearest other center (the scale at which a
                           center move shifts a Voronoi boundary)
    ``center_travel``      the same max displacement measured over the whole window (C_t versus
                           C_{t-W}) divided by the mean one-step drift over the window: about 1
                           when the centers jitter around a fixed point, about sqrt(W) for a
                           random walk and W for a steady drift
    stable when the drift averaged over the window is below ``center_tol`` (the centers have
    stopped), or when the drift has stopped decreasing (a noise floor), is below
    ``center_noise_tol`` and the centers do not travel (``center_travel <= travel_ratio``,
    they jitter around a fixed point); a noisy base algorithm whose centers still drift
    under the jitter is not stable yet

Partitions — how stable the clusters are
    ``cluster_unstable``   max over clusters of the share of its members still moving
    ``size_change``        max over clusters of the relative size change over the window
    ``net_flow``           net over gross transfers between pairs of clusters in the window
                           (0: every move from A to B is matched by one from B to A; 1: all
                           moves go one way)
    stable when no cluster changes size by more than ``size_tol`` and either few of its
    members move (``cluster_unstable <= cluster_tol``) or the moves balance out
    (``net_flow <= flow_tol``): no cluster is being reorganised, points only move across
    boundaries back and forth

Decision (``rule="auto"``)
    converged            no label changed (exact engines) -> stop, nothing to settle
    drift (tail)         unstable_frac <= unstable_tol -> stop and settle
    intervene            unstable_frac <= max_unstable, representatives and partitions stable,
                         and either the oscillation has reached a plateau (oscillating_share >=
                         osc_share and change counts no longer fall) -> regime "oscillation",
                         or the projected remaining changes <= change_tol -> regime "drift"
    otherwise            keep iterating
``rule="unstable"`` keeps only the converged / unstable_tol tests (the 0.1.0 behaviour).

Settlement (``settle``, ``rule="auto"``) depends on the regime the monitor diagnosed:

* oscillation — the remaining points keep moving between clusters, so every unstable point
  is settled by its window occupancy (soft: the occupancy, e.g. [0.5, 0.5]);
* drift — a point that switched at most once moved and stayed, so it keeps its current
  label; a point that switched twice or more is settled by its window occupancy
  (``oscillation_rule="frequency"``) or by the long-run law of its label Markov chain
  (``"markov"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _markov as _mk

STOP_RULES = ("auto", "unstable")
ASSIGN_RULES = ("auto", "markov", "frequency", "keep")


@dataclass(frozen=True)
class StopParams:
    window: int = 10
    min_iter: int = 10
    unstable_tol: float = 1e-3
    max_unstable: float = 0.2
    osc_share: float = 0.3
    plateau_ratio: float = 0.8
    change_tol: float = 1e-3
    center_tol: float = 1e-2
    center_noise_tol: float = 0.25
    cluster_tol: float = 0.1
    size_tol: float = 0.02
    travel_ratio: float = 2.0
    flow_tol: float = 0.1


def switches(Hw):
    """Number of label switches of each column of a (W+1, m) label window."""
    Hw = np.asarray(Hw)
    return (Hw[1:] != Hw[:-1]).sum(0)


def plateau(values, window, ratio):
    """True when ``values`` over the last ``window`` steps stopped decreasing."""
    c = np.asarray(values[-window:], dtype=float)
    if len(c) < window or window < 2:
        return False
    h = window // 2
    first, second = c[:h].mean(), c[h:].mean()
    return first > 0 and second >= ratio * first


def net_flow(Hw, k):
    """Net over gross transfers between cluster pairs in a (W+1, m) label window."""
    Hw = np.asarray(Hw, dtype=np.int64)
    a, b = Hw[:-1].ravel(), Hw[1:].ravel()
    moved = a != b
    if not moved.any():
        return 0.0
    pair = np.minimum(a[moved], b[moved]) * k + np.maximum(a[moved], b[moved])
    sign = np.where(a[moved] < b[moved], 1, -1)
    keys, inv = np.unique(pair, return_inverse=True)
    net = np.abs(np.bincount(inv, weights=sign, minlength=len(keys))).sum()
    return float(net / moved.sum())


def projected_changes(counts, window, n):
    """Remaining label changes (share of n) if the change counts keep decaying geometrically."""
    c = np.asarray(counts[-window:], dtype=float)
    if len(c) < window or window < 2:
        return float("inf")
    h = window // 2
    first, second = c[:h].mean(), c[h:].mean()
    if first <= 0:
        return 0.0
    if second <= 0:
        return 0.0
    q = (second / first) ** (1.0 / (window - h))
    if q >= 1.0:
        return float("inf")
    return float(c[-1] * q / (1.0 - q) / n)


def _half_nearest(C):
    k = C.shape[0]
    if k == 1:
        return np.full(1, np.inf)
    D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return 0.5 * D.min(1)


@dataclass
class StateMonitor:
    """Tracks the run iteration by iteration and decides when the Markov step intervenes."""

    n: int
    k: int
    params: StopParams = field(default_factory=StopParams)
    exact: bool = True
    rule: str = "auto"

    def __post_init__(self):
        W = self.params.window
        dtype = np.uint8 if self.k <= 255 else (np.uint16 if self.k <= 65535 else np.int32)
        self.H = np.empty((W + 1, self.n), dtype=dtype)   # ring buffer of the last W+1 labelings
        self.last_change = np.full(self.n, -1, dtype=np.int32)
        self.counts = []
        self.drift = []
        self.trace = []
        self.t = -1
        self._C = None
        self._Cring = None    # centers of the last W+1 iterations

    def push(self, labels, centers, changed=None):
        """Record iteration t = 0, 1, 2, ... (t = 0 is the initial assignment)."""
        W = self.params.window
        self.t += 1
        t = self.t
        lab = np.asarray(labels).astype(self.H.dtype, copy=False)
        if t > 0:
            moved = lab != self.H[(t - 1) % (W + 1)]
            self.last_change[moved] = t
            self.counts.append(int(moved.sum()) if changed is None else int(changed))
            step = np.sqrt(((centers - self._C) ** 2).sum(1))
            self.drift.append(float((step / _half_nearest(centers)).max()))
        self.H[t % (W + 1)] = lab
        self._C = np.array(centers, dtype=np.float64, copy=True)
        if self._Cring is None:
            self._Cring = np.empty((W + 1,) + self._C.shape)
        self._Cring[t % (W + 1)] = self._C

    def travel(self):
        """Max center displacement over the window relative to the mean one-step drift."""
        W, t = self.params.window, self.t
        if t < W or W < 2:
            return float("inf")
        step = float(np.mean(self.drift[-W:]))
        if step <= 0:
            return 0.0
        C = self._Cring[t % (W + 1)]
        move = np.sqrt(((C - self._Cring[(t - W) % (W + 1)]) ** 2).sum(1)) / _half_nearest(C)
        return float(move.max() / step)

    def window(self, idx, t=None):
        t = self.t if t is None else t
        W = self.params.window
        rows = [s % (W + 1) for s in range(max(0, t - W), t + 1)]
        return self.H[np.ix_(rows, idx)]

    def unstable(self):
        return np.flatnonzero(self.last_change > self.t - self.params.window)

    def assess(self):
        """Return (stop, regime, state) for the current iteration."""
        p, t, W = self.params, self.t, self.params.window
        changed = self.counts[-1] if self.counts else None
        state = {"t": t, "changed": changed}
        if self.exact and changed == 0:
            return self._log(True, "converged", state)
        if t < max(p.min_iter, W):
            return self._log(False, "running", state)
        u = float((self.last_change > t - W).mean())
        state["unstable_frac"] = u
        if u <= p.unstable_tol:
            return self._log(True, "drift", state)
        if self.rule != "auto" or u > p.max_unstable:
            return self._log(False, "running", state)
        # representatives
        d = self.drift[-1]
        d_mean = float(np.mean(self.drift[-W:]))
        travel = self.travel()
        reps = d_mean <= p.center_tol or (plateau(self.drift, W, p.plateau_ratio) and d <= p.center_noise_tol
                                          and travel <= p.travel_ratio)
        # partitions
        U = self.unstable()
        cur = self.H[t % (W + 1)].astype(np.int64)
        sizes = np.bincount(cur, minlength=self.k).astype(float)
        old = np.bincount(self.H[(t - W) % (W + 1)].astype(np.int64), minlength=self.k).astype(float)
        cl_unstable = float((np.bincount(cur[U], minlength=self.k) / np.maximum(sizes, 1)).max())
        size_change = float((np.abs(sizes - old) / np.maximum(old, 1)).max())
        Hw = self.window(U)
        flow = net_flow(Hw, self.k) if len(U) else 0.0
        parts = size_change <= p.size_tol and (cl_unstable <= p.cluster_tol or flow <= p.flow_tol)
        # movement
        osc = float((switches(Hw) >= 2).mean()) if len(U) else 0.0
        rho = projected_changes(self.counts, W, self.n)
        flat = plateau(self.counts, W, p.plateau_ratio)
        state.update(center_drift=d, center_drift_window=d_mean, center_travel=travel, representatives_stable=bool(reps), cluster_unstable=cl_unstable,
                     size_change=size_change, net_flow=flow, partitions_stable=bool(parts), oscillating_share=osc,
                     projected_changes=rho, counts_plateau=bool(flat))
        if reps and parts:
            if osc >= p.osc_share and flat:
                return self._log(True, "oscillation", state)
            if rho <= p.change_tol:
                return self._log(True, "drift", state)
        return self._log(False, "running", state)

    def _log(self, stop, regime, state):
        state["decision"] = regime if stop else "continue"
        self.trace.append(state)
        return stop, regime, state


def settle(Hw, current, rule="auto", alpha=0.0, pooled_prior=0.0, n_clusters=None, oscillation_rule="frequency",
           regime="drift"):
    """Settle the m unstable points from their (W+1, m) label windows.

    Returns (labels, states, probs, oscillating) where (states, probs) is the sparse soft
    membership (visited clusters padded with -1) and ``oscillating`` flags points that
    switched at least twice.
    """
    Hw = np.ascontiguousarray(Hw, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    m = Hw.shape[1]
    osc = switches(Hw) >= 2
    if rule == "keep":
        S = np.full((m, Hw.shape[0]), -1, dtype=np.int64)
        P = np.zeros((m, Hw.shape[0]))
        S[:, 0] = current
        P[:, 0] = 1.0
        return current.copy(), S, P, osc
    if rule == "auto" and regime == "oscillation":
        S, P = _mk.frequency(Hw)
        labels = _mk.argmax_label(S, P, current) if m else current.copy()
        return labels, S, P, osc
    if rule == "markov" or (rule == "auto" and oscillation_rule == "markov"):
        prior = _mk.pooled_prior(Hw, int(n_clusters), pooled_prior) if pooled_prior > 0 else None
        S, P = _mk.stationary(Hw, alpha=alpha, prior=prior)
    elif rule in ("frequency", "auto"):
        S, P = _mk.frequency(Hw)
    else:
        raise ValueError(f"assign_rule must be one of {ASSIGN_RULES}")
    if rule == "auto":
        S = S.copy()
        P = P.copy()
        keep = ~osc
        S[keep] = -1
        P[keep] = 0.0
        S[keep, 0] = current[keep]
        P[keep, 0] = 1.0
    labels = _mk.argmax_label(S, P, current) if m else current.copy()
    return labels, S, P, osc
