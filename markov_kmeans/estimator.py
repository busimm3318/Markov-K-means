"""MarkovKMeans: stop K-means once only a few points still move, settle them by Markov chains."""
from __future__ import annotations

import time

import numpy as np

from kmeans_accel import markov_assign as _mk
from kmeans_accel.core import as_data, compute_centers, inertia
from kmeans_accel.tail import soft_centers

from ._engines import HamerlyEngine, LloydEngine, MiniBatchEngine

_ALGORITHMS = ("hamerly", "lloyd", "minibatch")


class MarkovKMeans:
    """Markov-K-means clustering.

    The base algorithm (exact Lloyd via Hamerly bounds by default, plain Lloyd, or
    mini-batch) runs while the last ``window + 1`` labels of every point are kept in a
    ring buffer.  It stops

    * at iteration ``t0`` when ``t0`` is given, or
    * as soon as the *unstable set* U — points whose label changed within the last
      ``window`` iterations — is at most ``unstable_tol`` of the data (checked from
      iteration ``max(min_iter, window)`` on), or
    * when the base algorithm converges / ``max_iter`` is reached.

    Each point of U is then settled by the Markov chain of its own label window: the
    transition matrix is built from the observed moves (optionally Dirichlet-smoothed by
    ``alpha`` and shrunk towards pooled moves by ``pooled_prior``) and the point goes to
    the cluster with the largest long-run probability.  The long-run distribution is
    also kept as a soft membership (``membership_``), e.g. ``[0.5, 0.5]`` for a point
    that alternated between two clusters.

    Parameters
    ----------
    n_clusters : int
    algorithm : {"hamerly", "lloyd", "minibatch"}
    init : {"k-means++", "random"} or array of shape (n_clusters, n_features)
    t0 : int or None
        Fixed stopping iteration; ``None`` uses the adaptive rule with ``unstable_tol``.
    unstable_tol : float
        Adaptive rule: stop once |U| / n <= unstable_tol.
    min_iter, max_iter : int
    window : int
        Length W of the label window that defines U and feeds the chains.
    alpha : float
        Dirichlet smoothing of transition counts (0 keeps absorbing states).
    pooled_prior : float
        Strength of the empirical-Bayes prior pooled over all unstable points.
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
    unstable_states_, unstable_probs_ : (|U|, window+1) visited clusters and long-run
        probabilities of each point of U (states padded with -1)
    uncertainty_ : (|U|,) 1 - max long-run probability
    membership_ : scipy.sparse CSR (n, n_clusters); one-hot outside U, long-run law on U
    n_iter_, stop_reason_, unstable_fraction_, inertia_, n_dist_, fit_seconds_
    """

    def __init__(self, n_clusters=8, *, algorithm="hamerly", init="k-means++", t0=None,
                 unstable_tol=1e-3, min_iter=10, max_iter=300, window=10, alpha=0.0,
                 pooled_prior=0.0, center_update="auto", batch_size=4096, learning_rate="count",
                 eta=2e-4, random_state=None):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.init = init
        self.t0 = t0
        self.unstable_tol = unstable_tol
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
        engine = self._engine(X, self._initial_centers(X))
        dtype = np.uint8 if k <= 255 else (np.uint16 if k <= 65535 else np.int32)
        H = np.empty((W + 1, n), dtype=dtype)       # ring buffer of the last W+1 labelings
        H[0] = engine.labels
        last_change = np.full(n, -1, dtype=np.int32)
        t, stop_reason, frac = 0, "max_iter", None
        while t < self.max_iter:
            changed = engine.step()
            t += 1
            lab = engine.labels.astype(dtype)
            moved = lab != H[(t - 1) % (W + 1)]
            last_change[moved] = t
            H[t % (W + 1)] = lab
            if changed == 0 and engine.exact:
                stop_reason = "converged"
                break
            if self.t0 is not None:
                if t >= self.t0:
                    stop_reason = "t0"
                    break
            elif t >= max(self.min_iter, W):
                frac = float((last_change > t - W).mean())
                if frac <= self.unstable_tol:
                    stop_reason = "few_unstable"
                    break

        L = engine.labels.copy()
        C = engine.centers
        if stop_reason == "converged":
            U = np.empty(0, dtype=np.int64)
            S = np.empty((0, W + 1), dtype=np.int64)
            P = np.empty((0, W + 1))
            labels = L
        else:
            U = np.flatnonzero(last_change > t - W).astype(np.int64)
            rows = [s % (W + 1) for s in range(max(0, t - W), t + 1)]
            Hw = np.ascontiguousarray(H[rows][:, U], dtype=np.int64)
            prior = _mk.pooled_prior(Hw, k, self.pooled_prior) if self.pooled_prior > 0 else None
            S, P = _mk.stationary(Hw, alpha=self.alpha, prior=prior)
            labels = L.copy()
            if len(U):
                labels[U] = _mk.argmax_label(S, P, L[U])

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
