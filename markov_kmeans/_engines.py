"""Base iteration engines driven one iteration at a time by :class:`MarkovKMeans`.

Each engine exposes ``labels`` (int64, current assignment), ``centers`` (the centers
those labels were computed against), ``n_dist`` (point-center distance evaluations so
far) and ``step() -> int`` (one iteration; returns the number of labels that changed).
"""
from __future__ import annotations

import numpy as np

from ._kernels import (_assign, _half_nearest_center, _init, _sculley_update, _update_bounds,
                       center_drift, compute_centers, nearest_all)


class LloydEngine:
    """Plain Lloyd: recompute centers, then assign every point (n k distances)."""

    exact = True

    def __init__(self, X, C0):
        self.X = X
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        n, k = X.shape[0], self.centers.shape[0]
        self.labels = np.full(n, -1, dtype=np.int64)
        nearest_all(X, self.centers, self.labels)
        self.n_dist = n * k

    def step(self):
        self.centers, _ = compute_centers(self.X, self.labels, self.centers)
        changed = nearest_all(self.X, self.centers, self.labels)
        self.n_dist += self.X.shape[0] * self.centers.shape[0]
        return int(changed)


class HamerlyEngine:
    """Hamerly (SDM 2010) bounds: identical iterates to Lloyd, far fewer distances."""

    exact = True

    def __init__(self, X, C0):
        self.X = X
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        n, k = X.shape[0], self.centers.shape[0]
        self.labels = np.empty(n, dtype=np.int64)
        self.u = np.empty(n)
        self.l = np.empty(n)
        _init(X, self.centers, self.labels, self.u, self.l)
        self.n_dist = n * k

    def step(self):
        new_C, _ = compute_centers(self.X, self.labels, self.centers)
        drift = center_drift(self.centers, new_C)
        self.centers = new_C
        s = _half_nearest_center(new_C)
        _update_bounds(self.labels, self.u, self.l, drift)
        changed, nd = _assign(self.X, new_C, self.labels, self.u, self.l, s)
        self.n_dist += int(nd)
        return int(changed)


class MiniBatchEngine:
    """Sculley (WWW 2010) mini-batch K-means; one step = one epoch over a random permutation.

    A point's label for the epoch is the one it receives when its batch is processed, so
    the label history costs nothing extra.  ``learning_rate="count"`` is Sculley's 1/count
    rule; ``"constant"`` uses a fixed per-point step ``eta`` (centers keep jittering).
    """

    exact = False

    def __init__(self, X, C0, batch_size=4096, learning_rate="count", eta=2e-4, seed=None):
        self.X = X
        self.centers = np.array(C0, dtype=np.float64, copy=True)
        n, k = X.shape[0], self.centers.shape[0]
        self.batch_size = int(batch_size)
        self.eta = float(eta) if learning_rate == "constant" else 0.0
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(k)
        self.x_sq = np.einsum("ij,ij->i", X, X)
        self.labels = np.empty(n, dtype=np.int64)
        for s in range(0, n, 1 << 16):
            idx = np.arange(s, min(n, s + (1 << 16)))
            self.labels[idx] = self._nearest(idx)
        self.n_dist = n * k

    def _nearest(self, idx):
        C = self.centers
        D = self.X[idx] @ C.T
        D *= -2.0
        D += np.einsum("ij,ij->i", C, C)
        D += self.x_sq[idx, None]
        return D.argmin(1)

    def step(self):
        prev = self.labels.copy()
        perm = self.rng.permutation(self.X.shape[0])
        for s in range(0, len(perm), self.batch_size):
            idx = perm[s:s + self.batch_size]
            lab = self._nearest(idx)
            self.labels[idx] = lab
            _sculley_update(self.X[idx], lab, self.centers, self.counts, self.eta)
        self.n_dist += self.X.shape[0] * self.centers.shape[0]
        return int((self.labels != prev).sum())
