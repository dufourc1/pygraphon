from pygraphon.estimators.BaseEstimator import BaseEstimator
import numpy as np


class ISFE(BaseEstimator):
    """Iterative step-function estimator :cite:p:`cai2014`."""

    def __init__(self) -> None:
        super().__init__()

    def _approximate_graphon_from_adjacency(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_initial_partition(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Get the initial partition of the graph."""
        raise NotImplementedError