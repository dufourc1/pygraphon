"""Implementation of USVD estimator."""
from typing import Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_graph import edge_density


class USVT(BaseEstimator):
    """Estimate graphons  via Universal Singular Value Thresholding [1].

    Parameters
    ----------
    eta : float
        Threshold for singular values. Must be between 0 and 1.


    References
    ----------
    [1] Chatterjee, Sourav. "Matrix estimation by universal singular value thresholding."
        The Annals of Statistics 43.1 (2015): 177-214.
    [2] Xu, Jiaming. "Rates of convergence of spectral methods for graphon estimation."
        International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(self, eta: float = 0.01) -> None:
        if eta < 0 or eta > 1:
            raise ValueError("eta must be between 0 and 1")
        self.eta = eta

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:
        rho = edge_density(adjacency_matrix)
        if rho == 0:
            raise ValueError("empty graph, cannot estimate graphon")

        threshold = (2 + self.eta) * np.sqrt(adjacency_matrix.shape[0])

        # compute the singular value decomposition and threshold
        u, s, v = np.linalg.svd(adjacency_matrix, full_matrices=False)
        s[s < threshold] = 0

        # reconstruct
        P_hat = u @ np.diag(s) @ v

        # clip values to [0,1]
        np.clip(P_hat, a_min=0, a_max=1, out=P_hat)

        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat
