"""Base class for graphon estimators."""
from abc import abstractclassmethod
from typing import Tuple, Union

import networkx as nx
import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


class BaseEstimator:
    """Base class for Graphon estimator."""

    def __init__(self) -> None:
        self.fitted = False
        self.graphon = None
        self.edge_connectivity = None

    def fit(self, graph: Union[nx.Graph, np.ndarray], *args, **kwargs):
        """Estimate the graphon function f(x,y) values from a realized graph or adjacency matrix.

        Parameters
        ----------
        graph : Union[nx.Graph, np.ndarray]
            networkx simple graph or adjacency matrix
        args: optional
            additional arguments
        kwargs: optional
            additional arguments

        Raises
        ------
        ValueError
            type of graph is not supported
        ValueError
            if both a graph and an adjacency matrix are provided and do not agree
        """
        if isinstance(graph, nx.Graph):
            adjacency_matrix = get_adjacency_matrix_from_graph(graph)
        elif isinstance(graph, np.ndarray):
            adjacency_matrix = graph
        else:
            raise ValueError(
                f"type of graph is not supported, got {type(graph)}, but expected nx.Graph or np.ndarray"
            )
        self.graphon, self.edge_connectivity = self._approximate_graphon_from_adjacency(
            adjacency_matrix, *args, **kwargs
        )
        self.fitted = True

    @abstractclassmethod
    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, *args, **kwargs
    ) -> Tuple[Graphon, np.ndarray]:
        """Estimate the graphon function f(x,y) from an adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjacency matrix
        args: optional
            additional arguments
        kwargs: optional
            additional arguments

        Returns
        -------
        Tuple[Graphon, np.ndarray]
            approximated graphon and matrix of connection Pij of size n x n
        """

    def get_graphon(self) -> Graphon:
        """Return the estimated graphon if available.

        If model is not fitted or graphon is not available, returns None.

        Returns
        -------
        Graphon
            graphon
        """
        return self.graphon

    def get_edge_connectivity(self) -> np.ndarray:
        """Return the estimated edge connectivity if available.

        If model is not fitted or graphon is not available, returns None.

        Returns
        -------
        np.ndarray
            edge connectivity
        """
        return self.edge_connectivity
