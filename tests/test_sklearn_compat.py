"""MarkovKMeans is a scikit-learn estimator: the library's own compatibility checks pass."""
import numpy as np
import pytest
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from markov_kmeans import MarkovKMeans

SK_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])

ESTIMATORS = [
    MarkovKMeans(),
    MarkovKMeans(algorithm="lloyd", t0=3, assign_rule="markov", center_update="soft"),
    MarkovKMeans(algorithm="minibatch", batch_size=16),
    MarkovKMeans(algorithm="minibatch", batch_size=16, learning_rate="constant", eta=0.05),
]


@parametrize_with_checks(ESTIMATORS)
def test_scikit_learn_checks(estimator, check):
    name = getattr(check, "func", check).__name__
    if (SK_VERSION < (1, 4) and name == "check_estimators_pickle"
            and getattr(check, "keywords", {}).get("readonly_memmap")):
        pytest.skip("scikit-learn < 1.4 test helper fails here for KMeans as well")
    check(estimator)


def test_pipeline_grid_search_and_pandas_output():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(60, 3)) for c in (0.0, 3.0, 6.0)])
    pipe = make_pipeline(StandardScaler(), MarkovKMeans(3, random_state=0))
    assert pipe.fit(X).predict(X).shape == (180,)
    search = GridSearchCV(MarkovKMeans(random_state=0), {"n_clusters": [2, 3, 4]}, cv=3).fit(X)
    assert search.best_params_["n_clusters"] in (2, 3, 4)
    pd = pytest.importorskip("pandas")
    out = MarkovKMeans(3, random_state=0).set_output(transform="pandas").fit_transform(pd.DataFrame(X))
    assert list(out.columns) == ["markovkmeans0", "markovkmeans1", "markovkmeans2"]


def test_random_state_instance_is_accepted():
    X = np.random.default_rng(1).normal(size=(200, 2))
    a = MarkovKMeans(4, random_state=np.random.RandomState(5)).fit(X)
    assert a.labels_.shape == (200,)
    b = MarkovKMeans(4, random_state=7).fit(X)
    c = MarkovKMeans(4, random_state=7).fit(X)
    np.testing.assert_array_equal(b.labels_, c.labels_)
