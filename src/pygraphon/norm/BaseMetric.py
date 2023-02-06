from abc import abstractclassmethod

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon


class BaseMetric:
    @abstractclassmethod
    def __call__(self, graphon: Graphon, estimator: Graphon):
        pass

    @abstractclassmethod
    def __str__(self):
        pass


class MSE_P_hat(BaseMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def __call__(self, graphon: Graphon, estimator: BaseEstimator):
        p_0 = graphon._get_edge_probabilities(self.n_nodes, exchangeable=False)
        if not estimator.fitted:
            raise ValueError("Estimator is not fitted")
        p_hat = estimator.graphon._get_edge_probabilities(self.n_nodes, exchangeable=False)
        return np.mean((p_0 - p_hat) ** 2)

    def __str__(self):
        return "MSE on probabily matrix"
