import numpy as np
import pytest

from kmeans_accel import datasets
from kmeans_accel.lloyd import lloyd
from markov_kmeans import MarkovKMeans, assign_from_history


@pytest.fixture(scope="module")
def blobs():
    return datasets.blobs(4000, 6, 12, sep=1.2, seed=3)


@pytest.mark.parametrize("algorithm", ["hamerly", "lloyd"])
def test_runs_to_convergence_equals_lloyd(blobs, algorithm):
    C0 = datasets.init_random(blobs, 12, seed=1)
    ref = lloyd(blobs, C0, max_iter=1000)
    mk = MarkovKMeans(12, algorithm=algorithm, init=C0, stop_rule="unstable", unstable_tol=0.0,
                      max_iter=1000).fit(blobs)
    assert mk.stop_reason_ == "converged"
    assert len(mk.unstable_indices_) == 0
    np.testing.assert_array_equal(mk.labels_, ref.labels)
    np.testing.assert_allclose(mk.cluster_centers_, ref.centers)
    assert mk.n_iter_ == ref.n_iter


def test_fixed_t0_settles_exactly_the_unstable_set(blobs):
    C0 = datasets.init_random(blobs, 12, seed=1)
    mk = MarkovKMeans(12, init=C0, t0=6, window=4, assign_rule="markov").fit(blobs)
    assert mk.stop_reason_ == "t0" and mk.n_iter_ == 6
    stable = np.ones(len(blobs), bool)
    stable[mk.unstable_indices_] = False
    np.testing.assert_array_equal(mk.labels_[stable], mk.labels_at_stop_[stable])
    # the stop labels are Lloyd's labels after t0 iterations
    ref = lloyd(blobs, C0, max_iter=6)
    np.testing.assert_array_equal(mk.labels_at_stop_, ref.labels)
    M = mk.membership_
    assert M.shape == (len(blobs), 12)
    np.testing.assert_allclose(np.asarray(M.sum(1)).ravel(), 1.0)
    np.testing.assert_array_equal(np.asarray(M.argmax(1)).ravel()[stable], mk.labels_[stable])
    assert np.all((mk.uncertainty_ >= 0) & (mk.uncertainty_ < 1))


def test_adaptive_stop_rule(blobs):
    mk = MarkovKMeans(12, unstable_tol=0.02, min_iter=5, window=5, random_state=0).fit(blobs)
    if mk.stop_reason_ == "few_unstable":
        assert mk.unstable_fraction_ <= 0.02
        assert mk.n_iter_ >= 5
    assert mk.stop_reason_ in ("few_unstable", "converged", "oscillation")
    assert mk.n_dist_ > 0 and mk.inertia_ > 0


def test_minibatch_constant_step_is_deterministic(blobs):
    kw = dict(algorithm="minibatch", learning_rate="constant", eta=1e-3, t0=8, batch_size=512, random_state=4)
    a = MarkovKMeans(12, **kw).fit(blobs)
    b = MarkovKMeans(12, **kw).fit(blobs)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)


def test_soft_center_update_uses_memberships(blobs):
    C0 = datasets.init_random(blobs, 12, seed=2)
    hard = MarkovKMeans(12, init=C0, t0=5, window=5, center_update="hard").fit(blobs)
    soft = MarkovKMeans(12, init=C0, t0=5, window=5, center_update="soft").fit(blobs)
    np.testing.assert_array_equal(hard.labels_, soft.labels_)
    assert np.isfinite(soft.cluster_centers_).all()


def test_predict_and_transform(blobs):
    mk = MarkovKMeans(12, random_state=0).fit(blobs)
    D = mk.transform(blobs[:50])
    assert D.shape == (50, 12) and (D >= 0).all()
    np.testing.assert_array_equal(mk.predict(blobs[:50]), D.argmin(1))
    assert mk.score(blobs) <= 0


def test_assign_from_history_patterns():
    hist = np.array([[0, 0, 0, 0, 0, 1, 1, 1],      # drift: absorbed in 1
                     [0, 1, 0, 1, 0, 1, 0, 1],      # oscillation
                     [0, 0, 0, 0, 1, 0, 0, 0]]).T    # excursion
    r = assign_from_history(hist)
    assert list(r.labels) == [1, 1, 0]
    D = r.dense(2)
    np.testing.assert_allclose(D[0], [0, 1])
    np.testing.assert_allclose(D[1], [0.5, 0.5])
    assert r.uncertainty[0] == pytest.approx(0) and r.uncertainty[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        assign_from_history(hist, pooled_prior=1.0)


def test_input_validation(blobs):
    with pytest.raises(ValueError):
        MarkovKMeans(5000).fit(blobs)
    with pytest.raises(ValueError):
        MarkovKMeans(3, algorithm="nope").fit(blobs)
    with pytest.raises(RuntimeError):
        MarkovKMeans(3).predict(blobs)


def test_package_is_standalone(tmp_path):
    """The published wheel ships only markov_kmeans: it must not import the research package."""
    import subprocess
    import sys

    code = ("import sys, numpy as np, markov_kmeans; "
            "m = markov_kmeans.MarkovKMeans(3, t0=2, random_state=0).fit(np.random.default_rng(0).normal(size=(300, 2))); "
            "assert 'kmeans_accel' not in sys.modules, 'research package imported'; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


# ----------------------------------------------------------------------------- decision rules

from markov_kmeans import _rules  # noqa: E402


def _monitor(n=1000, k=2, **kw):
    params = _rules.StopParams(window=4, min_iter=4, **kw)
    return _rules.StateMonitor(n, k, params, exact=False)


def _run(mon, steps, labels_at, centers_at):
    out = None
    for t in range(steps + 1):
        mon.push(labels_at(t), centers_at(t))
        if t:
            out = mon.assess()
            if out[0]:
                return t, out
    return steps, out


def test_monitor_intervenes_on_stable_oscillation():
    base = np.r_[np.zeros(500, int), np.ones(500, int)]
    C = np.array([[0.0, 0.0], [10.0, 0.0]])

    def labels(t):          # 20 points (2%) swap clusters every iteration, the rest are fixed
        lab = base.copy()
        lab[:20] = t % 2
        return lab

    t, (stop, regime, state) = _run(_monitor(), 30, labels, lambda t: C)
    assert stop and regime == "oscillation", state
    assert state["representatives_stable"] and state["partitions_stable"]
    assert state["oscillating_share"] == 1.0


def test_monitor_waits_while_representatives_move():
    base = np.r_[np.zeros(500, int), np.ones(500, int)]

    def labels(t):
        lab = base.copy()
        lab[:20] = t % 2
        return lab

    def centers(t):          # center 0 travels a large, quickly shrinking distance
        return np.array([[3.0 * 0.7 ** t, 0.0], [10.0, 0.0]])

    mon = _monitor()
    t, (stop, regime, state) = _run(mon, 40, labels, centers)
    early = [s_ for s_ in mon.trace if "representatives_stable" in s_][:3]
    assert early and not any(s_["representatives_stable"] for s_ in early)
    assert all(s_["decision"] == "continue" for s_ in early)
    if stop:                 # intervention only once the centers have settled
        assert state["representatives_stable"] and state["center_drift"] <= mon.params.center_tol


def test_monitor_waits_while_a_cluster_reorganises():
    C = np.array([[0.0, 0.0], [10.0, 0.0]])

    def labels(t):           # 15% of cluster 0 keeps flipping: not boundary noise
        lab = np.r_[np.zeros(500, int), np.ones(500, int)]
        lab[:75] = t % 2
        return lab

    t, (stop, regime, state) = _run(_monitor(max_unstable=0.2), 30, labels, lambda t: C)
    assert not stop and not state["partitions_stable"] and state["cluster_unstable"] > 0.1, state


def test_monitor_tail_and_convergence():
    mon = _rules.StateMonitor(1000, 2, _rules.StopParams(window=4, min_iter=4), exact=True)
    lab = np.r_[np.zeros(500, int), np.ones(500, int)]
    C = np.array([[0.0, 0.0], [10.0, 0.0]])
    mon.push(lab, C)
    for t in range(1, 3):
        mon.push(lab, C, changed=0 if t == 2 else 1)
        stop, regime, _ = mon.assess()
    assert (stop, regime) == (True, "converged")
    assert _rules.projected_changes([100, 50, 25, 12], 4, 1000) < 0.02
    assert _rules.projected_changes([10, 10, 10, 10], 4, 1000) == float("inf")


def test_settle_auto_keeps_drifters_and_averages_oscillators():
    hist = np.array([[0, 0, 0, 0, 0, 1, 1, 1],      # drift: keep current label 1
                     [0, 1, 1, 1, 0, 0, 1, 1],      # 3 switches: occupancy 0.375 / 0.625
                     [2, 2, 2, 2, 2, 2, 2, 2]]).T
    labels, S, P, osc = _rules.settle(hist, hist[-1], rule="auto")
    assert list(labels) == [1, 1, 2]
    assert list(osc) == [False, True, False]
    from markov_kmeans._markov import to_dense
    D = to_dense(S, P, 3)
    np.testing.assert_allclose(D[0], [0, 1, 0])
    np.testing.assert_allclose(D[1], [3 / 8, 5 / 8, 0])
    lab_k, _, _, _ = _rules.settle(hist, hist[-1], rule="keep")
    np.testing.assert_array_equal(lab_k, hist[-1])


def test_auto_detects_persistent_oscillation():
    X = datasets.blobs(30000, 4, 8, sep=1.0, seed=5)
    mk = MarkovKMeans(8, algorithm="minibatch", learning_rate="constant", eta=1e-3, batch_size=1024,
                      max_iter=120, random_state=0).fit(X)
    assert mk.stop_reason_ == "oscillation", (mk.stop_reason_, mk.n_iter_, mk.unstable_fraction_)
    assert mk.regime_ == "oscillation" and mk.oscillating_share_ >= 0.3
    assert mk.n_iter_ < 120
    last = mk.decision_trace_[-1]
    assert last["decision"] == "oscillation" and last["representatives_stable"] and last["partitions_stable"]


def test_settle_in_oscillation_regime_uses_occupancy_for_every_point():
    hist = np.array([[0, 0, 0, 0, 0, 1, 1, 1],      # one switch: kept in drift, occupancy here
                     [0, 1, 0, 1, 0, 1, 0, 1]]).T
    labels, S, P, osc = _rules.settle(hist, hist[-1], rule="auto", regime="oscillation")
    from markov_kmeans._markov import to_dense
    D = to_dense(S, P, 2)
    np.testing.assert_allclose(D[0], [5 / 8, 3 / 8])
    assert list(labels) == [0, 1]
    labels_d, _, _, _ = _rules.settle(hist, hist[-1], rule="auto", regime="drift")
    assert list(labels_d) == [1, 1]


def test_monitor_waits_while_centers_travel_under_jitter():
    base = np.r_[np.zeros(500, int), np.ones(500, int)]

    def labels(t):
        lab = base.copy()
        lab[:20] = t % 2
        return lab

    def centers(drift):
        rng = np.random.default_rng(0)
        jitter = rng.normal(scale=0.04, size=(61, 2, 2))
        return lambda t: np.array([[0.0, 0.0], [10.0, 0.0]]) + jitter[t] + [[drift * t, 0.0], [0.0, 0.0]]

    params = _rules.StopParams(window=10, min_iter=10)
    # jitter at a noise floor, but center 0 keeps moving one way: not stable yet
    mon = _rules.StateMonitor(1000, 2, params, exact=False)
    t, (stop, regime, state) = _run(mon, 60, labels, centers(0.04))
    assert not stop, (t, state)
    assert state["center_drift"] <= params.center_noise_tol and state["center_travel"] > params.travel_ratio
    # the same jitter around fixed centers: stable, the oscillation is settled
    mon = _rules.StateMonitor(1000, 2, params, exact=False)
    t, (stop, regime, state) = _run(mon, 60, labels, centers(0.0))
    assert stop and regime == "oscillation", state
    assert state["center_travel"] <= params.travel_ratio
