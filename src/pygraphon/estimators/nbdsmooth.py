"""Implementation of neighborhood smoothing estimator."""
from pygraphon.estimators.BaseEstimator import BaseEstimator


class NBD(BaseEstimator):
    """Estimate graphon by neighborhood smoothing."""

    def _approximate_graphon_from_adjacency(self, adjacency_matrix):
        raise NotImplementedError()
