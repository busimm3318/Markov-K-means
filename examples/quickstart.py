"""Markov-K-means quick start: cluster, inspect the few unstable points, read soft memberships.

    python examples/quickstart.py
"""
import time

import numpy as np

from kmeans_accel import datasets
from markov_kmeans import MarkovKMeans, assign_from_history

X = datasets.blobs(1_000_000, 16, 50, sep=1.5, seed=0)
MarkovKMeans(5, t0=3).fit(X[:2000])     # warm up the numba kernels so timings are fair

t = time.perf_counter()
mk = MarkovKMeans(n_clusters=50, random_state=0).fit(X)   # stop_rule="auto": judges the run's state
print(f"stopped after {mk.n_iter_} iterations ({mk.stop_reason_}, regime {mk.regime_}) "
      f"in {time.perf_counter() - t:.1f}s")
last = mk.decision_trace_[-1]
print("  state at the stop: " + ", ".join(f"{k}={last[k]:.3g}" for k in ("unstable_frac", "center_drift")
                                           if isinstance(last.get(k), float)))
print(f"unstable points settled by Markov chains: {len(mk.unstable_indices_)} "
      f"({mk.unstable_fraction_:.3%} of the data)")

M = mk.membership_                      # sparse (n, k): one-hot outside U, long-run law on U
frac = np.flatnonzero(M.max(1).toarray().ravel() < 1)
print(f"points with a fractional membership: {len(frac)}")
for i in frac[:3]:
    row = M[i].toarray().ravel()
    print(f"  point {i}: " + ", ".join(f"cluster {j} -> {row[j]:.2f}" for j in np.flatnonzero(row)))

# Reference: run the same base algorithm to full convergence.
full = MarkovKMeans(n_clusters=50, stop_rule="unstable", unstable_tol=0.0, max_iter=10_000, random_state=0).fit(X)
# (Hamerly already makes late iterations cheap in distances; the saving is mostly passes over the data.)
print(f"full convergence: {full.n_iter_} iterations; labels differing from Markov-K-means: "
      f"{(full.labels_ != mk.labels_).mean():.3%}")
print(f"saved vs full run: {1 - mk.n_dist_ / full.n_dist_:.1%} of distance evaluations, "
      f"{1 - mk.fit_seconds_ / full.fit_seconds_:.1%} of fit time")

# A constant-step mini-batch (like an EMA-trained VQ codebook) never settles: boundary points keep
# oscillating. The monitor detects the stationary oscillation and settles them by their occupancy.
osc = MarkovKMeans(n_clusters=50, algorithm="minibatch", learning_rate="constant", eta=1e-3,
                   max_iter=200, random_state=0).fit(X)
print(f"constant step: stopped at epoch {osc.n_iter_} ({osc.stop_reason_}); "
      f"{osc.oscillating_share_:.0%} of its {len(osc.unstable_indices_)} unstable points oscillate")

# The Markov step on its own works on any label history (rows = iterations, oldest first).
history = np.array([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]).T
r = assign_from_history(history)
print("assign_from_history:", r.labels, r.dense(2).round(2).tolist())
