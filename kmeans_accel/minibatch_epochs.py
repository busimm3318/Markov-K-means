"""Mini-batch K-means (Sculley, "Web-scale k-means clustering", WWW 2010) run in epochs.

Each epoch visits every point exactly once in a random order, in batches of ``batch_size``.
A batch is assigned to the current centers (cached), then each point pulls its center
with the per-center learning rate 1 / count.  The label a point receives when its batch
is processed is its label for that epoch, so a per-point label history comes for free.

``lr="constant"`` replaces 1/count by a fixed ``eta`` (centers keep jittering: the
persistent-oscillation regime).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from .core import as_data


@njit(cache=True)
def _sculley_update(Xb, lab, C, counts, eta):
    d = C.shape[1]
    for r in range(Xb.shape[0]):
        c = lab[r]
        counts[c] += 1.0
        step = 1.0 / counts[c] if eta <= 0.0 else eta
        for t in range(d):
            C[c, t] += step * (Xb[r, t] - C[c, t])


def minibatch_epochs(X, init_centers, epochs, batch_size=4096, seed=0, lr="count", eta=0.0,
                     callback=None):
    """Run ``epochs`` passes; ``callback(e, labels, centers, n_dist, seconds)`` after each.

    ``labels`` passed to the callback are the labels assigned during epoch e (1-based);
    e = 0 is the nearest-center assignment of the initial centers.
    """
    X = as_data(X)
    C = np.array(init_centers, dtype=np.float64, copy=True)
    n, k = X.shape[0], C.shape[0]
    rng = np.random.default_rng(seed)
    counts = np.zeros(k)
    labels = np.empty(n, dtype=np.int64)
    x_sq = np.einsum("ij,ij->i", X, X)
    step_eta = eta if lr == "constant" else 0.0

    def assign(idx, out):
        D = X[idx] @ C.T
        D *= -2.0
        D += np.einsum("ij,ij->i", C, C)
        D += x_sq[idx, None]
        out[:] = D.argmin(1)

    t0 = time.perf_counter()
    for s in range(0, n, 1 << 16):
        assign(np.arange(s, min(n, s + (1 << 16))), labels[s:s + (1 << 16)])
    if callback is not None:
        callback(0, labels, C, n * k, time.perf_counter() - t0)
    lab_b = np.empty(batch_size, dtype=np.int64)
    for e in range(1, epochs + 1):
        te = time.perf_counter()
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            out = lab_b[:len(idx)]
            assign(idx, out)
            labels[idx] = out
            _sculley_update(X[idx], out, C, counts, step_eta)
        if callback is not None:
            callback(e, labels, C, n * k, time.perf_counter() - te)
    return C, labels
