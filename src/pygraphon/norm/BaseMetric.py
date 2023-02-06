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
        ajd_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if ajd_matrix is not None:
            estimator.fit(adj_matrix=ajd_matrix, **kwargs)
        if not estimator.fitted:
            raise ValueError("Estimator is not fitted and no adjacency matrix was provided")
        return self._compute(graphon=graphon, estimated=estimator.graphon)

    @abstractclassmethod
    def _compute(self, graphon: Graphon, estimated: Graphon):
        pass

    @abstractclassmethod
    def __str__(self):
        pass


class MSE_P_hat(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon):
        p_0 = graphon._get_edge_probabilities(self.n_nodes, exchangeable=False)
        p_hat = estimated._get_edge_probabilities(self.n_nodes, exchangeable=False)
        return np.mean((p_0 - p_hat) ** 2)

    def __str__(self):
        return "MSE on probabily matrix"
