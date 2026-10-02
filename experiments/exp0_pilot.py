"""Exp 0 (pilot): what do the 'non-converging' points look like after T0 Lloyd iterations?

For exact Lloyd (k-means++ init, run to zero label change) it records every point's
label history and reports
  * how many passes over the data happen after T0 (the "tail"),
  * how many points still change label after T0 and how (drift vs. oscillation),
  * whether instability observed *before* T0 predicts change *after* T0,
and for mini-batch k-means it reports how many points keep flipping once centers are
nearly stable.

    python -m experiments.exp0_pilot > experiments/results/exp0_pilot.txt
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

from kmeans_accel import datasets
from kmeans_accel.core import compute_centers

T0 = 10


def assign(X, xsq, C):
    return (xsq[:, None] - 2 * X @ C.T + (C * C).sum(1)).argmin(1)


def lloyd_history(X, k, seed, max_iter=1000):
    """H[t] = labels after t center updates; stops at the first unchanged assignment."""
    xsq = (X * X).sum(1)
    C, _ = kmeans_plusplus(X, k, random_state=seed)
    H = [assign(X, xsq, C)]
    for _ in range(max_iter):
        C, _ = compute_centers(X, H[-1], C)
        H.append(assign(X, xsq, C))
        if np.array_equal(H[-1], H[-2]):
            break
    return np.array(H)


def tail_report(name, X, k, seed):
    H = lloyd_history(X, k, seed)
    T = len(H) - 1
    after = H[T0:]
    will = (after != H[T0]).any(0)
    sub = after[:, will]
    n_states = np.array([len(np.unique(sub[:, j])) for j in range(sub.shape[1])], dtype=int)
    n_switch = (sub[1:] != sub[:-1]).sum(0)
    back = sub[-1] == sub[0]
    states = {int(s): int((n_states == s).sum()) for s in np.unique(n_states)}
    print(f"{name:20s} seed={seed} n={X.shape[0]:6d} d={X.shape[1]:4d} k={k:3d} | iters={T:4d} | "
          f"tail passes={T - T0:4d} ({(T - T0) / T:5.1%}) | changed after T0={will.sum():6d} "
          f"({will.mean():6.2%}) | final!=label@T0={(H[-1] != H[T0]).mean():6.2%} | "
          f"states visited {states} | median switches={np.median(n_switch):.0f} | "
          f"returned to label@T0={back.mean():.2f}")
    for W in (3, 5, 10):
        flagged = (H[T0 - W:T0 + 1] != H[T0]).any(0)
        tp = (flagged & will).sum()
        print(f"{'':20s}   flag = changed in last {W:2d} iters before T0: flagged={flagged.sum():6d} "
              f"recall={tp / max(will.sum(), 1):.2f} precision={tp / max(flagged.sum(), 1):.2f}")


def minibatch_report(name, X, k, seed=0, steps=600, every=10, burn_in=20):
    xsq = (X * X).sum(1)
    mb = MiniBatchKMeans(k, batch_size=1024, random_state=seed, n_init=1)
    rng = np.random.default_rng(seed)
    H, shifts, prev = [], [], None
    for step in range(steps):
        mb.partial_fit(X[rng.integers(0, len(X), 1024)])
        if step % every == every - 1:
            H.append(assign(X, xsq, mb.cluster_centers_))
            if prev is not None:
                shifts.append(np.linalg.norm(mb.cluster_centers_ - prev, axis=1).max())
            prev = mb.cluster_centers_.copy()
    H = np.array(H[burn_in:])
    flips = H[1:] != H[:-1]
    flipper = flips.any(0)
    sw = flips.sum(0)[flipper]
    print(f"{name:20s} checkpoints={len(H)} | ever flipping={flipper.sum()} ({flipper.mean():.2%}) | "
          f"flips per checkpoint (median)={np.median(flips.sum(1)):.0f} | switches per flipper "
          f"median={np.median(sw):.0f} max={sw.max()} | ended at first label={np.mean(H[-1][flipper] == H[0][flipper]):.2f} | "
          f"late max center shift={np.median(shifts[-20:]):.4f}")


def main():
    cases = [
        ("blobs overlap d=16", datasets.blobs(50000, 16, 50, sep=1.0, seed=0), 50),
        ("blobs separated d=16", datasets.blobs(50000, 16, 50, sep=3.0, seed=0), 50),
        ("uniform d=8", datasets.uniform(50000, 8, seed=0), 50),
        ("fashion_mnist 20k", datasets.fashion_mnist(n=20000), 50),
        ("birch3", datasets.birch(3), 100),
    ]
    print(f"== Exact Lloyd, k-means++ init, T0={T0}")
    for name, X, k in cases:
        for seed in (0, 1):
            tail_report(name, X, k, seed)
    print("== Mini-batch k-means (batch 1024, label snapshot every 10 batches, first 20 discarded)")
    for name, X, k in (cases[0], cases[3]):
        minibatch_report(name, X, k)


if __name__ == "__main__":
    main()
