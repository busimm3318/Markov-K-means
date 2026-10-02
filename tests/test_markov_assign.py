import numpy as np
import pytest

from kmeans_accel.markov_assign import argmax_label, frequency, pooled_prior, stationary, to_dense


def run(seq, **kw):
    S, P = stationary(np.array(seq)[:, None], **kw)
    return {int(s): p for s, p in zip(S[0], P[0]) if s >= 0}


def test_absorbing_drift_goes_to_new_state():
    # A A A A A B B B: B never left -> all mass on B (frequency would favour A).
    pi = run([0, 0, 0, 0, 0, 1, 1, 1])
    assert pi[1] == pytest.approx(1.0) and pi[0] == pytest.approx(0.0, abs=1e-12)
    S, F = frequency(np.array([0, 0, 0, 0, 0, 1, 1, 1])[:, None])
    assert F[0][list(S[0]).index(0)] == pytest.approx(5 / 8)


def test_periodic_oscillation_is_half_half():
    pi = run([0, 1, 0, 1, 0, 1])
    assert pi[0] == pytest.approx(0.5) and pi[1] == pytest.approx(0.5)


def test_two_state_closed_form():
    seq = [2, 2, 5, 2, 2, 2, 5, 5, 2, 5, 5]
    pi = run(seq)
    n = np.zeros((2, 2))
    m = {2: 0, 5: 1}
    for a, b in zip(seq[:-1], seq[1:]):
        n[m[a], m[b]] += 1
    p = n / n.sum(1, keepdims=True)
    expect0 = p[1, 0] / (p[0, 1] + p[1, 0])
    assert pi[2] == pytest.approx(expect0, rel=1e-9)


def test_smoothing_makes_chain_irreducible():
    pi = run([0, 0, 0, 1, 1, 1], alpha=1.0)
    # counts+1: row0 = [3, 2], row1 = [1, 3] -> pi0 = p10/(p01+p10)
    p01, p10 = 2 / 5, 1 / 4
    assert pi[0] == pytest.approx(p10 / (p01 + p10), rel=1e-9)


def test_three_states_and_stable_point():
    pi = run([0, 1, 2, 0, 1, 2, 0])
    assert sum(pi.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 3) for v in pi.values())
    assert run([4, 4, 4, 4]) == {4: pytest.approx(1.0)}


def test_vectorised_matches_single_and_sums_to_one():
    rng = np.random.default_rng(0)
    H = rng.integers(0, 4, size=(11, 300))
    S, P = stationary(H, alpha=0.5)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
    for i in (0, 17, 299):
        single = run(list(H[:, i]), alpha=0.5)
        got = {int(s): p for s, p in zip(S[i], P[i]) if s >= 0}
        assert got.keys() == single.keys()
        for key in got:
            assert got[key] == pytest.approx(single[key])


def test_argmax_ties_prefer_current_label_and_dense():
    S = np.array([[3, 1, -1], [3, 1, -1]])
    P = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    assert list(argmax_label(S, P, np.array([1, 7]))) == [1, 1]
    D = to_dense(S, P, 5)
    np.testing.assert_allclose(D[0], [0, 0.5, 0, 0.5, 0])


def test_pooled_prior_rows():
    H = np.array([[0, 0], [1, 0], [1, 1]])
    G = pooled_prior(H, 2, strength=2.0)
    np.testing.assert_allclose(G, [[2 / 3, 4 / 3], [0, 2]])
