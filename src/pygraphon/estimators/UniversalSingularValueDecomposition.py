"""Implementation of USVD estimator."""
from typing import Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_graph import edge_density


class USVT(BaseEstimator):
    r"""Universal Singular Value Thresholding :cite:p:`Chatterjee2015`.

    The user can choose between to thresholding method depending on the
    sparsitiy of the graph.


    Parameters
    ----------
    eta : float
        Parameter for threshold for singular values. Must be between 0 and 1 (default: 0.01).

    regime : str
        The regime of the graph. Must be either "dense" or "sparse". (default: "dense")


    ..  note::
        - Dense graphs:
            the threshold from  :cite:p:`Chatterjee2015` should be used by setting the
            parameter :py:obj:`regime` to "dense". The parameter :py:obj:`eta` (:math:`\eta`)
            can be any arbitrary small positive number. The threshold is given by
            :math:`(2 + \eta) * \sqrt{n}`.

        - Sparse graphs:
            the threshold from :cite:p:`xu2018rates` should be used by setting the parameter
            :py:obj:`regime` to "sparse". The parameter :math:`\delta` can be any small positive
            number. The threshold is given by :math:`(1+\delta)*\kappa* \sqrt(n* \rho)`, where
            :math:`\kappa=4` if :math:`n \rho=\omega(\log n)` and :math:`\kappa=2` if
            :math:`n \rho=\omega(\log ^4 n)`.
    """

    def __init__(self, eta: float = 0.01, regime: str = "dense") -> None:
        if eta < 0 or eta > 1:
            raise ValueError("eta must be between 0 and 1")
        if regime not in ["dense", "sparse"]:
            raise ValueError("regime must be either 'dense' or 'sparse'")
        self.eta = eta
        self.regime = regime

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:
        rho = edge_density(adjacency_matrix)
        if rho == 0:
            raise ValueError("empty graph, cannot estimate graphon")

        if self.regime == "dense":
            threshold = (2 + self.eta) * np.sqrt(adjacency_matrix.shape[0])
        else:
            kappa = 4 if rho * adjacency_matrix.shape[0] < np.log(adjacency_matrix.shape[0]) else 2
            threshold = (1 + self.eta) * kappa * np.sqrt(adjacency_matrix.shape[0] * rho)

        # compute the singular value decomposition and threshold
        u, s, v = np.linalg.svd(adjacency_matrix, full_matrices=False)
        s[s < threshold] = 0

        # reconstruct
        P_hat = u @ np.diag(s) @ v

        # clip values to [0,1]
        np.clip(P_hat, a_min=0, a_max=1, out=P_hat)

        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat
