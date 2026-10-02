# Changelog

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
