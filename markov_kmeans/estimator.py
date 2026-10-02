"""MarkovKMeans: stop K-means once only a few points still move, settle them by Markov chains."""
from __future__ import annotations

import time

import numpy as np

from . import _rules
from ._engines import HamerlyEngine, LloydEngine, MiniBatchEngine
from ._kernels import as_data, compute_centers, inertia, soft_centers

_ALGORITHMS = ("hamerly", "lloyd", "minibatch")


class MarkovKMeans:
    """Markov-K-means clustering.

    The base algorithm (exact Lloyd via Hamerly bounds by default, plain Lloyd, or
    mini-batch) runs while a :class:`markov_kmeans._rules.StateMonitor` judges, after each
    iteration, the state of the whole run (``stop_rule="auto"``):

    * movement — the share of vectors still changing cluster within the last ``window``
      iterations (the unstable set U), how many of them oscillate (switch twice or more),
      and how many label changes are still to come judging by the decay of the counts;
    * representatives — the largest center move relative to half the distance to the
      nearest other center, which must be small, or have settled at a noise floor with the
      centers jittering around fixed points rather than travelling;
    * partitions — no cluster may still be reorganising (share of its members moving,
      relative change of its size over the window).

    The Markov step intervenes (the run stops and U is settled from its label histories)
    when only a few vectors move and centers and partitions are stable, and either the
    oscillation has reached a plateau (``stop_reason_="oscillation"``) or the projected
    remaining changes are negligible (``"stable_drift"``); it also stops on convergence
    (``"converged"``), when U is at most ``unstable_tol`` of the data (``"few_unstable"``),
    at ``t0`` or at ``max_iter``.  ``decision_trace_`` records every judgement.

    The points of U are then settled from their label windows (``assign_rule="auto"``),
    depending on the diagnosed regime: in an oscillation every point of U gets its window
    occupancy (hard: most frequent label; soft: the occupancy, e.g. ``[0.5, 0.5]``); in a
    drift a point that switched at most once (it moved and stayed) keeps its current label
    and only points that switched twice or more get their occupancy.  ``assign_rule="markov"`` uses the long-run
    law of each point's Markov chain instead (optionally smoothed by ``alpha`` and shrunk
    towards pooled moves by ``pooled_prior``); ``"frequency"`` the occupancy for every
    point; ``"keep"`` the current label.

    Parameters
    ----------
    n_clusters : int
    algorithm : {"hamerly", "lloyd", "minibatch"}
    init : {"k-means++", "random"} or array of shape (n_clusters, n_features)
    t0 : int or None
        Fixed stopping iteration (overrides ``stop_rule``).
    stop_rule : {"auto", "unstable"}
        "unstable" only uses ``unstable_tol`` (the 0.1.0 behaviour).
    assign_rule : {"auto", "markov", "frequency", "keep"}
    unstable_tol, max_unstable, osc_share, plateau_ratio, change_tol : float
        Movement thresholds of the stop rule.
    center_tol, center_noise_tol, travel_ratio : float
        Representative stability: relative center drift, averaged over the window, below
        ``center_tol``; or flat, below ``center_noise_tol`` and with the displacement over the
        window at most ``travel_ratio`` times the mean one-step drift (jitter, not travel).
    cluster_tol, size_tol : float
        Partition stability: max share of a cluster's members still moving, max relative
        cluster-size change over the window.
    min_iter, max_iter : int
    window : int
        Length W of the label window that defines U and feeds the assignment.
    alpha, pooled_prior : float
        Markov rule only: Dirichlet smoothing and pooled-prior strength.
    oscillation_rule : {"frequency", "markov"}
        How ``assign_rule="auto"`` settles oscillating points.
    center_update : {"auto", "hard", "soft", "none"}
        Final centers: centroids of the final labels ("hard"), weighted by the soft
        memberships ("soft"), or the base algorithm's current centers ("none").
        "auto" is "hard" for Lloyd/Hamerly and "none" for mini-batch.
    batch_size, learning_rate, eta : mini-batch settings (``learning_rate`` is
        ``"count"`` for Sculley's 1/count rule or ``"constant"`` for a fixed ``eta``).
    random_state : int or None

    Attributes
    ----------
    labels_ : (n,) final hard labels
    cluster_centers_ : (n_clusters, n_features)
    labels_at_stop_ : (n,) labels of the base algorithm when it stopped
    unstable_indices_ : indices of U (empty if the base algorithm converged)
    unstable_states_, unstable_probs_ : (|U|, window+1) visited clusters and settlement
        probabilities of each point of U (states padded with -1)
    oscillating_ : (|U|,) bool, points of U that switched at least twice in the window
    uncertainty_ : (|U|,) 1 - max probability
    membership_ : scipy.sparse CSR (n, n_clusters); one-hot outside U, soft law on U
    stop_reason_ : "converged", "few_unstable", "stable_drift", "oscillation", "t0" or "max_iter"
    regime_ : "converged", "drift" or "oscillation"
    decision_trace_ : list of dicts, the state judged at every iteration (movement,
        representatives, partitions, decision)
    n_iter_, unstable_fraction_, oscillating_share_, change_counts_, center_drift_, inertia_,
    n_dist_, fit_seconds_
    """

    def __init__(self, n_clusters=8, *, algorithm="hamerly", init="k-means++", t0=None,
                 stop_rule="auto", assign_rule="auto", unstable_tol=1e-3, max_unstable=0.2,
                 osc_share=0.3, plateau_ratio=0.8, change_tol=1e-3, center_tol=1e-2,
                 center_noise_tol=5e-2, travel_ratio=2.0, cluster_tol=0.1, size_tol=0.02, min_iter=10,
                 max_iter=300, window=10,
                 alpha=0.0, pooled_prior=0.0, oscillation_rule="frequency", center_update="auto",
                 batch_size=4096, learning_rate="count", eta=2e-4, random_state=None):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.init = init
        self.t0 = t0
        self.stop_rule = stop_rule
        self.assign_rule = assign_rule
        self.unstable_tol = unstable_tol
        self.max_unstable = max_unstable
        self.osc_share = osc_share
        self.plateau_ratio = plateau_ratio
        self.change_tol = change_tol
        self.center_tol = center_tol
        self.center_noise_tol = center_noise_tol
        self.travel_ratio = travel_ratio
        self.cluster_tol = cluster_tol
        self.size_tol = size_tol
        self.oscillation_rule = oscillation_rule
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.window = window
        self.alpha = alpha
        self.pooled_prior = pooled_prior
        self.center_update = center_update
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta = eta
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting

    def _initial_centers(self, X):
        k = self.n_clusters
        if isinstance(self.init, str):
            if self.init == "k-means++":
                from sklearn.cluster import kmeans_plusplus

                seed = None if self.random_state is None else int(self.random_state)
                C, _ = kmeans_plusplus(X, k, random_state=seed)
                return C
            if self.init == "random":
                rng = np.random.default_rng(self.random_state)
                return X[rng.choice(X.shape[0], size=k, replace=False)].copy()
            raise ValueError(f"unknown init {self.init!r}")
        C = np.asarray(self.init, dtype=np.float64)
        if C.shape != (k, X.shape[1]):
            raise ValueError(f"init has shape {C.shape}, expected {(k, X.shape[1])}")
        return C

    def _engine(self, X, C0):
        if self.algorithm == "hamerly":
            return HamerlyEngine(X, C0)
        if self.algorithm == "lloyd":
            return LloydEngine(X, C0)
        if self.algorithm == "minibatch":
            return MiniBatchEngine(X, C0, self.batch_size, self.learning_rate, self.eta, self.random_state)
        raise ValueError(f"algorithm must be one of {_ALGORITHMS}")

    def fit(self, X, y=None):
        t_start = time.perf_counter()
        X = as_data(X)
        n, k, W = X.shape[0], self.n_clusters, int(self.window)
        if not 1 <= k <= n:
            raise ValueError("need 1 <= n_clusters <= n_samples")
        if W < 1:
            raise ValueError("window must be >= 1")
        if self.stop_rule not in _rules.STOP_RULES:
            raise ValueError(f"stop_rule must be one of {_rules.STOP_RULES}")
        if self.assign_rule not in _rules.ASSIGN_RULES:
            raise ValueError(f"assign_rule must be one of {_rules.ASSIGN_RULES}")
        params = _rules.StopParams(
            window=W, min_iter=self.min_iter, unstable_tol=self.unstable_tol,
            max_unstable=self.max_unstable, osc_share=self.osc_share, plateau_ratio=self.plateau_ratio,
            change_tol=self.change_tol, center_tol=self.center_tol, center_noise_tol=self.center_noise_tol,
            travel_ratio=self.travel_ratio, cluster_tol=self.cluster_tol, size_tol=self.size_tol)
        engine = self._engine(X, self._initial_centers(X))
        monitor = _rules.StateMonitor(n, k, params, exact=engine.exact, rule=self.stop_rule)
        monitor.push(engine.labels, engine.centers)

        t, stop_reason, regime = 0, "max_iter", "running"
        while t < self.max_iter:
            changed = engine.step()
            t += 1
            monitor.push(engine.labels, engine.centers, changed)
            if self.t0 is not None:
                if changed == 0 and engine.exact:
                    stop_reason = regime = "converged"
                    break
                if t >= self.t0:
                    stop_reason = "t0"
                    break
                continue
            stop, regime, _ = monitor.assess()
            if stop:
                stop_reason = {"converged": "converged", "drift": "few_unstable" if
                               monitor.trace[-1].get("unstable_frac", 1) <= self.unstable_tol else "stable_drift",
                               "oscillation": "oscillation"}[regime]
                break

        L = engine.labels.copy()
        C = engine.centers
        if stop_reason == "converged":
            U = np.empty(0, dtype=np.int64)
            S = np.empty((0, W + 1), dtype=np.int64)
            P = np.empty((0, W + 1))
            oscillating = np.empty(0, dtype=bool)
            labels = L
        else:
            U = monitor.unstable().astype(np.int64)
            Hw = monitor.window(U)
            if regime == "running":   # stopped by t0 or max_iter: classify from the window
                share = float((_rules.switches(Hw) >= 2).mean()) if len(U) else 0.0
                regime = "oscillation" if share >= self.osc_share else "drift"
            labels = L.copy()
            labels_U, S, P, oscillating = _rules.settle(
                Hw, L[U], rule=self.assign_rule, alpha=self.alpha, pooled_prior=self.pooled_prior,
                n_clusters=k, oscillation_rule=self.oscillation_rule, regime=regime)
            labels[U] = labels_U

        mode = self.center_update
        if mode == "auto":
            mode = "none" if self.algorithm == "minibatch" else "hard"
        if mode == "hard":
            centers = compute_centers(X, labels, C)[0]
        elif mode == "soft":
            centers = soft_centers(X, L, U, S, P, C) if len(U) else compute_centers(X, labels, C)[0]
        elif mode == "none":
            centers = C.copy()
        else:
            raise ValueError("center_update must be 'auto', 'hard', 'soft' or 'none'")

        self.labels_ = labels
        self.labels_at_stop_ = L
        self.cluster_centers_ = centers
        self.unstable_indices_ = U
        self.unstable_states_ = S
        self.unstable_probs_ = P
        self.uncertainty_ = 1.0 - P.max(1) if len(U) else np.empty(0)
        self.n_iter_ = t
        self.stop_reason_ = stop_reason
        self.regime_ = regime
        self.oscillating_ = oscillating
        self.oscillating_share_ = float(oscillating.mean()) if len(U) else 0.0
        self.change_counts_ = np.asarray(monitor.counts, dtype=np.int64)
        self.center_drift_ = np.asarray(monitor.drift)
        self.decision_trace_ = monitor.trace
        self.unstable_fraction_ = len(U) / n
        self.inertia_ = float(inertia(X, centers, labels))
        self.n_dist_ = int(engine.n_dist)
        self.n_features_in_ = X.shape[1]
        self._n = n
        self._membership = None
        self.fit_seconds_ = time.perf_counter() - t_start
        return self

    # ------------------------------------------------------------------ outputs

    @property
    def membership_(self):
        """Sparse (n, n_clusters) memberships: one-hot outside U, Markov long-run law on U."""
        if self._membership is None:
            from scipy import sparse

            n, k = self._n, self.n_clusters
            stable = np.ones(n, dtype=bool)
            stable[self.unstable_indices_] = False
            rows_s = np.flatnonzero(stable)
            ok = self.unstable_states_ >= 0
            rows_u = np.repeat(self.unstable_indices_, ok.sum(1))
            rows = np.concatenate([rows_s, rows_u])
            cols = np.concatenate([self.labels_[rows_s], self.unstable_states_[ok]])
            vals = np.concatenate([np.ones(len(rows_s)), self.unstable_probs_[ok]])
            self._membership = sparse.csr_matrix((vals, (rows, cols)), shape=(n, k))
        return self._membership

    def _check_fitted(self):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("call fit first")

    def transform(self, X, chunk=65536):
        """Euclidean distance of each row of X to each center."""
        self._check_fitted()
        X = as_data(X)
        C = self.cluster_centers_
        c_sq = np.einsum("ij,ij->i", C, C)
        out = np.empty((X.shape[0], C.shape[0]))
        for s in range(0, X.shape[0], chunk):
            xb = X[s:s + chunk]
            D = xb @ C.T
            D *= -2.0
            D += c_sq
            D += np.einsum("ij,ij->i", xb, xb)[:, None]
            np.sqrt(np.maximum(D, 0.0), out=out[s:s + chunk])
        return out

    def predict(self, X, chunk=65536):
        """Nearest learned center for new points (no label history, so no Markov step)."""
        self._check_fitted()
        X = as_data(X)
        out = np.empty(X.shape[0], dtype=np.int64)
        for s in range(0, X.shape[0], chunk):
            out[s:s + chunk] = self.transform(X[s:s + chunk]).argmin(1)
        return out

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def score(self, X, y=None):
        """Negative SSE of X to its nearest learned center (scikit-learn convention)."""
        D = self.transform(X)
        return -float((D.min(1) ** 2).sum())

    def __repr__(self):
        return (f"MarkovKMeans(n_clusters={self.n_clusters}, algorithm={self.algorithm!r}, "
                f"t0={self.t0}, unstable_tol={self.unstable_tol}, window={self.window})")
