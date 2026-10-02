"""Decision rules of Markov-K-means: when to stop, and how to settle the unstable points.

Both rules are pure functions of quantities the fit loop already tracks, so the
estimator and offline evaluations (experiments/auto_rule.py) share the exact same code.

Stop rule ("auto")
------------------
After each iteration t the run is classified from three global signals:

* ``u``  — unstable fraction: points whose label changed within the last W iterations;
* ``o``  — oscillating share: among those, the points that switched at least twice;
* the trend of the per-iteration change counts over the last W iterations.

==========================================================  ===========================
state                                                       decision
==========================================================  ===========================
no label changed (exact engines)                            ``converged``  (stop, exact)
``u <= unstable_tol``                                       ``drift``      (stop: tail)
``u <= max_unstable`` and ``o >= osc_share`` and counts      ``oscillation`` (stop: the
flat (second half of window >= plateau_ratio x first half)   remaining points never settle)
otherwise                                                   continue
==========================================================  ===========================

Assignment rule ("auto")
------------------------
A point that switched at most once in the window has an absorbing history (it moved and
stayed): it keeps its current label.  A point that switched twice or more is settled by
the label occupancy of its window (hard: most frequent, ties to the current label; soft:
the occupancy itself).  ``"markov"`` uses the Markov long-run law for every point,
``"frequency"`` the occupancy for every point, ``"keep"`` the current label.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def switches(Hw):
    """Number of label switches of each column of a (W+1, m) label window."""
    Hw = np.asarray(Hw)
    return (Hw[1:] != Hw[:-1]).sum(0)


def plateau(change_counts, window, ratio):
    """True when the change counts of the last ``window`` iterations stopped decreasing."""
    c = np.asarray(change_counts[-window:], dtype=float)
    if len(c) < window or window < 2:
        return False
    h = window // 2
    first, second = c[:h].mean(), c[h:].mean()
    return first > 0 and second >= ratio * first


def decide_stop(t, changed, unstable_frac, oscillating_share, change_counts, params, exact,
                rule="auto"):
    """Return (stop, regime) after iteration t.

    ``oscillating_share`` may be ``None`` when it was not computed (it is only needed once
    ``unstable_frac <= max_unstable``).
    """
    if exact and changed == 0:
        return True, "converged"
    if t < max(params.min_iter, params.window):
        return False, "running"
    if unstable_frac <= params.unstable_tol:
        return True, "drift"
    if rule == "auto" and unstable_frac <= params.max_unstable and oscillating_share is not None:
        if oscillating_share >= params.osc_share and plateau(change_counts, params.window, params.plateau_ratio):
            return True, "oscillation"
    return False, "running"


def settle(Hw, current, rule="auto", alpha=0.0, pooled_prior=0.0, n_clusters=None, oscillation_rule="frequency"):
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
