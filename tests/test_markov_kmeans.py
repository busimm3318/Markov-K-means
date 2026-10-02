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
    mk = MarkovKMeans(12, algorithm=algorithm, init=C0, unstable_tol=0.0, max_iter=1000).fit(blobs)
    assert mk.stop_reason_ == "converged"
    assert len(mk.unstable_indices_) == 0
    np.testing.assert_array_equal(mk.labels_, ref.labels)
    np.testing.assert_allclose(mk.cluster_centers_, ref.centers)
    assert mk.n_iter_ == ref.n_iter


def test_fixed_t0_settles_exactly_the_unstable_set(blobs):
    C0 = datasets.init_random(blobs, 12, seed=1)
    mk = MarkovKMeans(12, init=C0, t0=6, window=4).fit(blobs)
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
    assert mk.stop_reason_ in ("few_unstable", "converged")
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
