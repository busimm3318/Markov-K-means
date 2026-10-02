"""Markov-K-means quick start: cluster, inspect the few unstable points, read soft memberships.

    python examples/quickstart.py
"""
import time

import numpy as np

from kmeans_accel import datasets
from markov_kmeans import MarkovKMeans, assign_from_history

X = datasets.blobs(200_000, 16, 50, sep=1.5, seed=0)

t = time.perf_counter()
mk = MarkovKMeans(n_clusters=50, unstable_tol=1e-3, random_state=0).fit(X)
print(f"stopped after {mk.n_iter_} iterations ({mk.stop_reason_}) in {time.perf_counter() - t:.1f}s")
print(f"unstable points settled by Markov chains: {len(mk.unstable_indices_)} "
      f"({mk.unstable_fraction_:.3%} of the data)")

M = mk.membership_                      # sparse (n, k): one-hot outside U, long-run law on U
frac = np.flatnonzero(M.max(1).toarray().ravel() < 1)
print(f"points with a fractional membership: {len(frac)}")
for i in frac[:3]:
    row = M[i].toarray().ravel()
    print(f"  point {i}: " + ", ".join(f"cluster {j} -> {row[j]:.2f}" for j in np.flatnonzero(row)))

# Reference: run the same base algorithm to full convergence.
full = MarkovKMeans(n_clusters=50, unstable_tol=0.0, max_iter=10_000, random_state=0).fit(X)
print(f"full convergence: {full.n_iter_} iterations; labels differing from Markov-K-means: "
      f"{(full.labels_ != mk.labels_).mean():.3%}")
print(f"saved vs full run: {1 - mk.n_dist_ / full.n_dist_:.1%} of distance evaluations, "
      f"{1 - mk.fit_seconds_ / full.fit_seconds_:.1%} of fit time "
      "(Hamerly makes late iterations cheap in distances, not in passes over the data)")

# The Markov step on its own works on any label history (rows = iterations, oldest first).
history = np.array([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]).T
r = assign_from_history(history)
print("assign_from_history:", r.labels, r.dense(2).round(2).tolist())
