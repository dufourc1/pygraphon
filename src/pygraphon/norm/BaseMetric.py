from abc import abstractclassmethod
from typing import Optional

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class BaseMetric:
    """Base class for metrics.

    Parameters
    ----------
        graphon : Graphon
            graphon that was used to generate the adjacency matrix
        estimator : BaseEstimator
            estimator that was used to estimate the graphon
        adj_matrix : np.ndarray, optional
            adjacency matrix of the graphon, by default None (used if the estimator is not fitted or if the metric
            relies on the adjacency matrix directly)
    """

    def __call__(
        self,
        graphon: Graphon,
        estimator: BaseEstimator,
        adj_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        if not estimator.fitted:
            raise ValueError("Estimator is not fitted and no adjacency matrix was provided")
        return self._compute(
            graphon=graphon, estimated=estimator.graphon, adjacency_matrix=adj_matrix
        )

    @abstractclassmethod
    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray) -> float:
        pass

    @abstractclassmethod
    def __str__(self):
        pass
