# Changelog

## Unreleased

### Changed
* **Parallel assignment.** The Hamerly and Lloyd assignment passes, the bound updates and
  the brute-force nearest-center pass run on all cores (numba `prange`). Each point's label
  and bounds depend on that point alone and the center sums stay serial, so results are
  identical to the serial kernels and do not depend on the number of threads (checked bit
  for bit against 0.3.1 on 108 configurations). On one Sentinel-2 tile at 60 m (3.3 million
  pixels) an iteration is about 3.6 times faster on four cores for k = 256 and k = 1024.

## 0.4.0

The Markov-K-means algorithm is now one flow: the state monitor stops the run, the
**Markov chain of each center** gives a pseudo representative, every point is assigned to
its nearest pseudo representative, and the **Markov chains of the labels** settle the points
that remain undecided.

### Added
* **Pseudo representatives** (`representatives="markov"`, the new default). At the stop,
  the path of each center over the window is fitted as an AR(1) chain
  c_{s+1} = a + rho c_s; its long-run mean a / (1 - rho) is where a converging center is
  heading, or the point a jittering center hovers around. Every point is assigned to its
  nearest pseudo representative (one extra assignment pass, pruned by the Hamerly bounds of
  the run). This moves the points that are about to cross a boundary, which no label
  history can see because they have not moved yet.
  * Safeguard: the pseudo labelling is kept only if its SSE (with its own centroids) is not
    above that of the history settlement; otherwise the result is exactly the 0.3 result.
  * The undecided points are now the unstable set plus the predicted movers. A point that
    oscillates keeps its history law; a predicted mover gets {new: 1 - f/2, old: f/2}, f
    being how late the boundary crosses it on the way to the pseudo representatives.
  * New parameters `representatives` (`"markov"` or `"current"`) and `rho_max`; new
    `center_update="pseudo"` (final centers = pseudo representatives).
  * New attributes `pseudo_centers_`, `center_rho_`, `pseudo_accepted_`, `predicted_moves_`.
  * `_rules.long_run_centers`, `_rules.crossing_fraction`, `_rules.undecided` and
    `StateMonitor.centers_window`.

### Changed
* `unstable_indices_`, `unstable_states_`, `unstable_probs_`, `oscillating_`,
  `uncertainty_`, `unstable_fraction_` and `membership_` cover the undecided points (the
  unstable set plus the predicted movers). The unstable share that stopped the run is in
  `decision_trace_[-1]["unstable_frac"]`.
* `center_update="auto"` on mini-batch uses the pseudo representatives when the pseudo
  labelling was kept (the current centers otherwise, as before).
* `n_dist_` includes the extra assignment pass: pruned by the run's Hamerly bounds on the
  default engine (a few percent of n x n_clusters distances), n x n_clusters on Lloyd and
  mini-batch. The pass reuses the stopped engine's arrays, so it allocates nothing n-sized.
* `representatives="current"` reproduces 0.3.1 exactly (checked bit for bit on 108
  configurations of algorithm, stop, settlement rule and center update).

### Evaluation
* Sentinel-2 (docs/results.md, section 9): on the twelve scenes of thirty million pixels,
  at the same stop as 0.3, the labels differing from the converged map fell in every scene
  (median 0.20 % to 0.07 %, worst 0.49 % to 0.35 %), for 0.4 to 1.2 % of the full run's time.
  The pseudo labelling was kept in all twelve scenes, and almost all the pixels it moved
  ended where the converged run puts them.
* Synthetic trajectories (section 10): the error at the automatic stop fell in all ten
  cases (exact Lloyd 6 to 22 %, 1/count mini-batch 14 to 17 %, constant step 3 to 15 %),
  for at most 1.6 % of the full run's time. Under a constant step the pseudo
  representatives halve the distance of the centers to the reference centers.
* `rho_max = 0.95` is best or tied on exact Lloyd; higher caps overshoot or are rejected by
  the safeguard.

## 0.3.1

### Changed
* **Faster state monitor on large data.** Recording the labels, stamping the points that
  moved, counting cluster sizes and finding the recently moved points are now single
  compiled passes instead of several n-sized numpy passes (the cluster-size count alone
  took over a second per judgement at thirty million points). The unstable share is
  counted without building the index list unless the partition test needs it.
  Decisions are unchanged: identical traces on all ten cached trajectories of the
  automatic-rule study. On Sentinel-2 scenes of thirty million pixels the monitor's cost
  per iteration fell from about half a second to about a tenth.
* Package author metadata: Dongsu Shim.

### Evaluation
* Real satellite images (docs/results.md, section 9): twelve Sentinel-2 scenes of about
  thirty million pixels each. The automatic rule stopped within 0.5 % of the converged
  partition in every scene without tuning, where the remote-sensing default (fewer than
  2 % changed or 10 iterations) differed by a median of 23 % and scikit-learn's default by
  up to 34 %.

## 0.3.0

Development status is now **Beta**: parameter names, defaults and fitted attributes are
kept stable; any change to them goes through a deprecation warning first.

### Added
* `MarkovKMeans` is a scikit-learn estimator (clusterer and transformer). It passes
  `sklearn.utils.estimator_checks.check_estimator` for scikit-learn 1.3 to 1.9 and works
  with `clone`, `Pipeline`, `GridSearchCV` and `set_output(transform="pandas")`.
  * Input is validated by scikit-learn: NaN, infinite, sparse, complex and empty inputs
    raise clear errors; `n_features_in_` and `feature_names_in_` are recorded, and
    `predict`, `transform` and `score` check them.
  * `get_feature_names_out` names the distance columns `markovkmeans0`, `markovkmeans1`, ….
  * `random_state` also accepts a `RandomState` instance; integer seeds give the same
    results as before.
* `CITATION.cff`: GitHub shows "Cite this repository".

### Changed
* Calling `predict`, `transform` or `score` before `fit` raises scikit-learn's
  `NotFittedError` (a subclass of `ValueError` and `AttributeError`) instead of
  `RuntimeError`.
* `n_clusters` larger than the number of samples raises
  `ValueError("n_samples=... should be >= n_clusters=...")`.
* `repr` follows scikit-learn: it shows the parameters that differ from their defaults.

## 0.2.0

### Added
* **Automatic judgement of the run** (`stop_rule="auto"`, the new default). After every
  iteration a `StateMonitor` judges three groups of signals:
  * movement: share of points still changing (`unstable_frac`), share of them that
    oscillate, projected remaining changes;
  * representatives: center drift relative to half the distance to the nearest other
    center, and `center_travel`, which separates jitter around fixed points from a drift;
  * partitions: cluster-size change, share of a cluster's members moving, and `net_flow`,
    which separates balanced back-and-forth moves from a reorganisation.
  The Markov step intervenes when few points move and centers and partitions are stable.
  New stop reasons are `"stable_drift"` and `"oscillation"`, and every judgement is kept
  in `decision_trace_`.
* **Regime-aware settlement** (`assign_rule="auto"`, the new default):
  * after a drift, points that switched once keep their label and oscillating points get
    their window occupancy;
  * after an oscillation, every unstable point gets its occupancy.
  `"markov"`, `"frequency"` and `"keep"` remain available.
* New parameters: `max_unstable`, `osc_share`, `plateau_ratio`, `change_tol`,
  `center_tol`, `center_noise_tol`, `travel_ratio`, `cluster_tol`, `size_tol`,
  `flow_tol`, `oscillation_rule`.
* New attributes: `regime_`, `oscillating_`, `oscillating_share_`, `change_counts_`,
  `center_drift_`, `decision_trace_`.

### Changed
* The 0.1.0 behaviour (stop when |U|/n <= `unstable_tol`, Markov settlement) is still
  available with `stop_rule="unstable", assign_rule="markov"`.

### Evaluation
* On 10 instances at n = 1e6 (exact Lloyd, 1/count and constant-step mini-batch) the
  automatic rule uses 6–83% of the full run's time at an error of 0.05–0.62%.
  * With a constant step, scikit-learn's tolerance rule and the 0.1.0 rule never stop.
    The automatic rule stops at 10–62% of the run with an error close to that of the
    full run.
  * See docs/results.md, section 6.

## 0.1.0

* First release: `MarkovKMeans` (Hamerly, Lloyd or mini-batch base algorithm; stop when
  the unstable set is small; Markov settlement of the unstable points) and
  `assign_from_history`.
