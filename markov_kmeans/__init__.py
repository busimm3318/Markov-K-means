"""Markov-K-means.

K-means that watches the state of the whole run (how many points still move, how stable
the representatives and the partition are), stops once only a few points keep changing
cluster, and settles those points from their own label histories: the long-run law of
their label Markov chain for points that drifted, their occupancy for points that keep
oscillating (hard label = most likely cluster, soft membership = the whole distribution).

    >>> from markov_kmeans import MarkovKMeans
    >>> mk = MarkovKMeans(n_clusters=50, random_state=0).fit(X)
    >>> mk.labels_, mk.cluster_centers_, mk.membership_

See README.md for the method, the evaluation (docs/results.md) and its limits.
"""
from .assign import HistoryAssignment, assign_from_history
from .estimator import MarkovKMeans

__all__ = ["MarkovKMeans", "assign_from_history", "HistoryAssignment"]
__version__ = "0.3.0"
