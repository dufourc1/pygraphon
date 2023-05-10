from abc import abstractclassmethod
from typing import Optional

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class BaseMetric:
    def __call__(
        self,
        graphon: Graphon,
        estimator: BaseEstimator,
        adj_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if not estimator.fitted:
            raise ValueError("Estimator is not fitted and no adjacency matrix was provided")
        return self._compute(
            graphon=graphon,
            adjacency_matrix=adj_matrix,
            estimated=estimator.get_graphon(),
            pij=estimator.get_edge_connectivity(),
        )

    @abstractclassmethod
    def _compute(
        self,
        graphon: Graphon,
        adjacency_matrix: np.ndarray,
        estimated: Graphon,
        pij: np.ndarray,
    ):
        pass

    @abstractclassmethod
    def __str__(self):
        pass
