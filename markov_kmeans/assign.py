"""Markov-chain assignment from label histories — usable with any iterative clustering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _markov as _mk


@dataclass
class HistoryAssignment:
    """Result of :func:`assign_from_history` for m points.

    ``states[i, :]`` are the clusters point i visited (padded with -1) and ``probs[i, :]``
    their long-run probabilities; ``labels`` is the hard choice (ties go to the current
    label) and ``uncertainty = 1 - max prob``.
    """

    labels: np.ndarray
    states: np.ndarray
    probs: np.ndarray
    uncertainty: np.ndarray

    def dense(self, n_clusters):
        """(m, n_clusters) membership matrix."""
        return _mk.to_dense(self.states, self.probs, n_clusters)


def assign_from_history(history, *, alpha=0.0, pooled_prior=0.0, n_clusters=None, current=None):
    """Settle points from their label histories with the long-run law of a Markov chain.

    Parameters
    ----------
    history : array of shape (n_steps, m)
        Cluster label of each of m points at consecutive iterations, oldest first.
    alpha : float
        Dirichlet smoothing added to every transition count among visited clusters.
        0 (default) keeps absorbing states: a point that moved A -> B and stayed is
        assigned to B with probability 1.
    pooled_prior : float
        Strength (pseudo-transitions per row) of an empirical-Bayes prior built from the
        moves of all m points; needs ``n_clusters``.
    n_clusters : int, optional
        Number of clusters (required with ``pooled_prior``).
    current : array of shape (m,), optional
        Current labels used to break ties; defaults to the last row of ``history``.

    Returns
    -------
    HistoryAssignment
    """
    H = np.asarray(history)
    if H.ndim != 2 or H.shape[0] < 2:
        raise ValueError("history must have shape (n_steps >= 2, m)")
    H = np.ascontiguousarray(H, dtype=np.int64)
    prior = None
    if pooled_prior > 0:
        if n_clusters is None:
            raise ValueError("pooled_prior needs n_clusters")
        prior = _mk.pooled_prior(H, int(n_clusters), pooled_prior)
    S, P = _mk.stationary(H, alpha=alpha, prior=prior)
    cur = H[-1] if current is None else np.asarray(current, dtype=np.int64)
    labels = _mk.argmax_label(S, P, cur)
    return HistoryAssignment(labels, S, P, 1.0 - P.max(1))
