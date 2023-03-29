"""Implementation of the matrix completion scheme estimator."""
from typing import Optional, Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


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

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[Graphon, np.ndarray]:
        raise NotImplementedError()
