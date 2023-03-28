from abc import abstractclassmethod
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
        if not estimator.fitted:
            raise ValueError("Estimator is not fitted and no adjacency matrix was provided")
        return self._compute(
            graphon=graphon, estimated=estimator.graphon, adjacency_matrix=ajd_matrix
        )

    @abstractclassmethod
    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        pass

    @abstractclassmethod
    def __str__(self):
        pass


class MseProbaEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        p_0 = graphon._get_edge_probabilities(self.n_nodes, exchangeable=False)
        p_hat = estimated._get_edge_probabilities(self.n_nodes, exchangeable=False)
        return np.mean((p_0 - p_hat) ** 2)

    def __str__(self):
        return "MSE on probabily matrix"


class AUCEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        p_hat = estimated._get_edge_probabilities(self.n_nodes, exchangeable=False)
        if not adjacency_matrix.shape == p_hat.shape:
            raise ValueError("Adjacency matrix and probabilities matrix have different shapes")
        return roc_auc_score(adjacency_matrix, p_hat)

    def __str__(self):
        return "AUC on probabily matrix"


class AUPRCEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        p_hat = estimated._get_edge_probabilities(self.n_nodes, exchangeable=False)
        if not adjacency_matrix.shape == p_hat.shape:
            raise ValueError("Adjacency matrix and probabilities matrix have different shapes")
        return average_precision_score(adjacency_matrix, p_hat)

    def __str__(self):
        return "AUPRC on probabily matrix"
