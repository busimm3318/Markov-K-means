"""Markov-K-means.

K-means that stops once only a few points are still changing cluster, and settles those
points with the long-run distribution of the Markov chain of their own label history
(hard label = most likely cluster, soft membership = the whole distribution).

    >>> from markov_kmeans import MarkovKMeans
    >>> mk = MarkovKMeans(n_clusters=50, random_state=0).fit(X)
    >>> mk.labels_, mk.cluster_centers_, mk.membership_

See README.md for the method, the evaluation (docs/results.md) and its limits.
"""
from .assign import HistoryAssignment, assign_from_history
from .estimator import MarkovKMeans

__all__ = ["MarkovKMeans", "assign_from_history", "HistoryAssignment"]
__version__ = "0.1.0"
