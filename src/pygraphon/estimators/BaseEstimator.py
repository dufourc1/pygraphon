"""Base class for graphon estimators."""
from abc import abstractclassmethod
from typing import Tuple, Union

import networkx as nx
import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils.utils_graph import (
    check_simple_adjacency_matrix,
    get_adjacency_matrix_from_graph,
)


class BaseEstimator:
    """Base class for Graphon estimator.

    If the method can do function estimation, `.get_graphon()` should return a Graphon object,
    otherwise it should return None.
    If the method can do value estimation, `.get_edge_connectivity()` should return a numpy array,
    otherwise it should return None.

    Notes
    -----
    For methods that can do value estimation, the function estimation is done by
    approximating the graphon with a step function with bandwidth 1/n, where n is the number of nodes.


    Examples
    --------
    >>> import networkx as nx
    >>> from pygraphon.estimators import USVT
    >>> estimator = USVT()
    >>> graph = nx.erdos_renyi_graph(n=100, p=0.1)
    >>> estimator.fit(graph)
    >>> graphon = estimator.get_graphon() # returns a stepgraphon
    >>> edge_connectivity = estimator.get_edge_connectivity() # returns a matrix 100x100
    """

    def __init__(self) -> None:
        self.fitted = False
        self.graphon: Graphon = None
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
            if the graph is not a simple graph
        """
        if isinstance(graph, nx.Graph):
            adjacency_matrix = get_adjacency_matrix_from_graph(graph)
        elif isinstance(graph, np.ndarray):
            adjacency_matrix = graph
        else:
            raise ValueError(
                f"type of graph is not supported, got {type(graph)}, but expected nx.Graph or np.ndarray"
            )
        check_simple_adjacency_matrix(adjacency_matrix)
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
