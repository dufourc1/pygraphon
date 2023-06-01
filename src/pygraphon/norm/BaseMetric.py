from abc import abstractclassmethod

import numpy as np

from pygraphon.graphons import Graphon
from pygraphon.utils.utils_graph import check_simple_adjacency_matrix


class FunctionMetric:
    def __call__(
        self,
        graphon: Graphon,
        graphon_2: Graphon,
        **kwargs,
    ) -> float:
        if not isinstance(graphon, Graphon) or not isinstance(graphon_2, Graphon):
            raise TypeError("graphon and graphon_2 must be Graphon objects")
        return self._compute(graphon, graphon_2, **kwargs)

    @abstractclassmethod
    def _compute(self, graphon: Graphon, graphon_2: Graphon, **kwargs) -> float:
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        pass


class ValueMetric:
    def __call__(self, adjacency_matrix: np.ndarray, pij: np.ndarray, **kwargs) -> float:
        if adjacency_matrix.shape != pij.shape:
            raise ValueError("adjacency_matrix and pij must have the same shape")
        check_simple_adjacency_matrix(adjacency_matrix)
        return self._compute(adjacency_matrix, pij, **kwargs)

    @abstractclassmethod
    def _compute(self, adj_matrix: np.ndarray, pij: np.ndarray, **kwargs) -> float:
        pass

    @abstractclassmethod
    def __str__(self) -> str:
        pass
