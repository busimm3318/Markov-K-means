"""Every registered exact method must reproduce Lloyd bit-for-bit from the same seeds."""
import numpy as np
import pytest

from kmeans_accel.lloyd import lloyd
from kmeans_accel.registry import by_kind, discover

EXACT = by_kind("exact")
GEMM = by_kind("exact-gemm")


def test_all_modules_import():
    _, errors = discover()
    assert not errors, errors


@pytest.mark.parametrize("name", sorted(EXACT))
def test_exact_matches_lloyd(name, case):
    _, X, k, C0 = case
    ref = lloyd(X, C0, max_iter=200)
    res = EXACT[name]["fn"](X, C0, max_iter=200)
    assert res.n_iter == ref.n_iter
    assert res.converged == ref.converged
    np.testing.assert_array_equal(res.labels, ref.labels)
    np.testing.assert_array_equal(res.centers, ref.centers)
    assert res.inertia == pytest.approx(ref.inertia, rel=1e-12)
    assert 0 < res.n_dist <= ref.n_dist


@pytest.mark.parametrize("name", sorted(EXACT))
def test_exact_respects_max_iter(name, case):
    _, X, k, C0 = case
    ref = lloyd(X, C0, max_iter=2)
    res = EXACT[name]["fn"](X, C0, max_iter=2)
    assert res.n_iter == ref.n_iter
    np.testing.assert_array_equal(res.labels, ref.labels)
    np.testing.assert_array_equal(res.centers, ref.centers)


@pytest.mark.parametrize("name", sorted(GEMM))
def test_gemm_close_to_lloyd(name, case):
    _, X, k, C0 = case
    ref = lloyd(X, C0, max_iter=200)
    res = GEMM[name]["fn"](X, C0, max_iter=200)
    assert res.inertia == pytest.approx(ref.inertia, rel=1e-6)
