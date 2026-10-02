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


def test_decide_stop_states():
    p = _rules.StopParams(window=4, min_iter=4, unstable_tol=1e-3, max_unstable=0.05,
                          osc_share=0.5, plateau_ratio=0.8)
    assert _rules.decide_stop(7, 0, 0.0, None, [5, 3, 1, 0], p, exact=True) == (True, "converged")
    assert _rules.decide_stop(2, 9, 0.5, None, [9, 9], p, exact=True) == (False, "running")
    assert _rules.decide_stop(9, 3, 5e-4, None, [9] * 9, p, exact=True) == (True, "drift")
    flat = [100, 90, 80, 85, 80, 82, 79, 81, 80]
    assert _rules.decide_stop(9, 80, 0.01, 0.9, flat, p, exact=False) == (True, "oscillation")
    falling = [100, 90, 80, 60, 40, 30, 20, 15, 10]
    assert _rules.decide_stop(9, 10, 0.01, 0.9, falling, p, exact=False) == (False, "running")
    assert _rules.decide_stop(9, 80, 0.01, 0.2, flat, p, exact=False) == (False, "running")
    assert _rules.decide_stop(9, 80, 0.01, 0.9, flat, p, exact=False, rule="unstable") == (False, "running")


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
