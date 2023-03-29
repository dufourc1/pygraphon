"""Implementation of the matrix completion scheme estimator."""
from typing import Optional, Tuple

import numpy as np

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
        progress: bool = False,
        adjust: bool = True,
    ) -> None:
        self.rank = rank
        self.tol = tol
        self.iternumber = iternumber
        self.progress = progress
        self.adjust = adjust
        self.solver = OptSpace(n_components=rank, tol=tol, max_iterations=iternumber)

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:
        # solve using OptSpace
        U, S, V = self.solver.solve(adjacency_matrix)
        P_hat = U.dot(S).dot(V.T)

        # adjust P_hat to be a valid probability matrix
        np.clip(P_hat, 0, 1, out=P_hat)
        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat
