import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pygraphon.graphons import Graphon
from pygraphon.norm.BaseMetric import BaseMetric


class AUCEdge(BaseMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if "n_nodes" in kwargs:
            warnings.warn("n_nodes is deprecated and has no effect", stacklevel=2)

    def _compute(
        self,
        graphon: Graphon,
        estimated: Graphon,
        adjacency_matrix: np.ndarray,
    ):
        indices = np.triu_indices(adjacency_matrix.shape[0], k=1)
        p_hat = estimated.get_edge_probabilities(adjacency_matrix.shape[0], exchangeable=False)
        return roc_auc_score(adjacency_matrix[indices], p_hat[indices])

    def __str__(self):
        return "AUC on probabily matrix"


class AUPRCEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(
        self,
        graphon: Graphon,
        estimated: Graphon,
        adjacency_matrix: np.ndarray,
    ):
        p_hat = estimated.get_edge_probabilities(adjacency_matrix.shape[0], exchangeable=False)
        indices = np.triu_indices(adjacency_matrix.shape[0], k=1)
        return average_precision_score(adjacency_matrix[indices], p_hat[indices])

    def __str__(self):
        return "AUPRC on probabily matrix"
