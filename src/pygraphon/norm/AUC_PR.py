import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pygraphon.graphons import Graphon
from pygraphon.norm.BaseMetric import BaseMetric


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
