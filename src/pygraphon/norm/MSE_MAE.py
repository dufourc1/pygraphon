import numpy as np

from pygraphon.graphons import Graphon
from pygraphon.norm.BaseMetric import BaseMetric


class MseProbaEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        p_0 = graphon.get_edge_probabilities(self.n_nodes, exchangeable=False)
        p_hat = estimated.get_edge_probabilities(self.n_nodes, exchangeable=False)
        return np.mean((p_0 - p_hat) ** 2)

    def __str__(self):
        return "MSE on probabily matrix"


class MaeProbaEdge(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, estimated: Graphon, adjacency_matrix: np.ndarray):
        p_0 = graphon.get_edge_probabilities(self.n_nodes, exchangeable=False)
        p_hat = estimated.get_edge_probabilities(self.n_nodes, exchangeable=False)
        return np.mean(np.abs(p_0 - p_hat))

    def __str__(self):
        return "MAE on probabily matrix"
