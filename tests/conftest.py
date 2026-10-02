import numpy as np
import pytest

from kmeans_accel import datasets


def _cases():
    # (id, X, k) — small enough for fast tests, varied in d, k and clusteredness.
    return [
        ("blobs-d2-k8", datasets.blobs(1500, 2, 8, sep=6.0, seed=1), 8),
        ("blobs-d8-k20", datasets.blobs(2000, 8, 20, sep=3.0, seed=2), 20),
        ("blobs-d32-k50", datasets.blobs(2500, 32, 30, sep=1.5, seed=3), 50),
        ("uniform-d3-k16", datasets.uniform(2000, 3, seed=4), 16),
        ("uniform-d16-k5", datasets.uniform(1500, 16, seed=5), 5),
        ("digits-k10", datasets.digits(), 10),
        ("blobs-d4-k1", datasets.blobs(500, 4, 3, seed=6), 1),
        ("blobs-d2-k200", datasets.blobs(3000, 2, 50, sep=8.0, seed=7), 200),
    ]


CASES = _cases()


@pytest.fixture(params=CASES, ids=[c[0] for c in CASES])
def case(request):
    name, X, k = request.param
    return name, X, k, datasets.init_random(X, k, seed=11)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
