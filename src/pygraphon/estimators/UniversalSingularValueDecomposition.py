"""Implementation of USVD estimator."""
import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator


class USVT(BaseEstimator):
    """Estimate graphons  via Universal Singular Value Thresholding.

    Parameters
    ----------
    eta : float
        Threshold for singular values. Must be between 0 and 1.
    """

    def __init__(self, eta: float = 0.01) -> None:
        if eta < 0 or eta > 1:
            raise ValueError("eta must be between 0 and 1")
        self.eta = eta

    def _approximate_graphon_from_adjacency(self, adjacency_matrix: np.ndarray, *args, **kwargs):
        raise NotImplementedError()
