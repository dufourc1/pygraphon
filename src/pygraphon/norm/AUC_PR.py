import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .BaseMetric import ValueMetric


class AUCEdge(ValueMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if "n_nodes" in kwargs:
            warnings.warn("n_nodes is deprecated and has no effect", stacklevel=2)

    def _compute(
        self,
        adjacency_matrix: np.ndarray,
        pij: np.ndarray,
        **kwargs,
    ):
        indices = np.triu_indices(adjacency_matrix.shape[0], k=1)
        return roc_auc_score(adjacency_matrix[indices], pij[indices])

    def __str__(self):
        return "AUC on probability matrix"


class AUPRCEdge(ValueMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if "n_nodes" in kwargs:
            warnings.warn("n_nodes is deprecated and has no effect", stacklevel=2)

    def _compute(
        self,
        adjacency_matrix: np.ndarray,
        pij: np.ndarray,
        **kwargs,
    ):
        indices = np.triu_indices(adjacency_matrix.shape[0], k=1)
        return average_precision_score(adjacency_matrix[indices], pij[indices])

    def __str__(self):
        return "AUPRC on probability matrix"
