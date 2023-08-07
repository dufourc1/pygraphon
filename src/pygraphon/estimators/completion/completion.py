"""Implementation of the matrix completion scheme estimator."""
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon

from .optspace import OptSpace


class Completion(BaseEstimator):
    """Estimate graphons via matrix completion scheme :cite:p:`Keshavan2010`."""

    def __init__(
        self,
        rank: Optional[int] = None,
        tol: float = 1e-3,
        iternumber: int = 20,
    ) -> None:
        self.rank = rank
        self.tol = tol
        self.iternumber = iternumber
        self.solver = OptSpace(n_components=rank, tol=tol, max_iterations=iternumber, sign=1)

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:
        # solve using OptSpace
        U, S, V = self.solver.solve(2 * (adjacency_matrix - 0.5))
        P_hat = ((U @ S @ V.T) + 1) / 2
        P_hat[np.diag_indices_from(P_hat)] = 0

        # adjust P_hat to be a valid probability matrix
        np.clip(P_hat, 0, 1, out=P_hat)

        if not np.allclose(P_hat, P_hat.T):
            logger.warning(
                f"P_hat is not symmetric on  {np.where(P_hat != P_hat.T)[0].shape} indices"
            )
            P_hat_new = (P_hat + P_hat.T) / 2
            logger.debug(
                f"mean of absolute difference with corrected P_hat {np.abs(P_hat - P_hat_new).mean()}"
            )
            P_hat = P_hat_new

        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat
