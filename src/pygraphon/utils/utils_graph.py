"""Random utils to deal with graphs."""
from collections import Counter

import networkx as nx
import numpy as np

from pygraphon.utils.utils_matrix import check_symmetric


def get_adjacency_matrix_from_graph(graph: nx.Graph) -> np.ndarray:
    """Get the adjacency matrix from a networkx graph.

    Parameters
    ----------
    graph : nx.Graph
        graph

    Returns
    -------
    np.ndarray
        adjacency matrix
    """
    return np.array(nx.adjacency_matrix(graph).todense())


def edge_density(adjacency_matrix: np.ndarray) -> float:
    """Compute the edge density of a sinmple graph represented by an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
         adjacency matrix representing the graph

    Returns
    -------
    float
        edge density rho
    """
    return np.sum(adjacency_matrix) / (adjacency_matrix.shape[0] * (adjacency_matrix.shape[1] - 1))


def check_simple_adjacency_matrix(adjacency_matrix: np.ndarray) -> None:
    """Check if an adjacency matrix is symmetric, binary and has no element on the diagonal.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
         matrix representing the graph to check.

    Raises
    ------
    ValueError
        if the adjacency matrix is not binary
    ValueError
        if the adjacency matrix is not symmetric
    ValueError
        if the adjacency matrix has elements on the diagonal
    ValueError
        if the adjacency matrix is not square
    ValueError
        if the adjacency matrix is not 2D
    ValueError
        if the adjacency matrix is not at least 2x2
    """
    if len(adjacency_matrix.shape) != 2:
        raise ValueError("Adjacency matrix should be 2D")
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix should be square")
    if adjacency_matrix.shape[0] < 2:
        raise ValueError("Adjacency matrix should be of dimension at least 2 x 2")
    if np.sum(np.diag(adjacency_matrix)) != 0:
        raise ValueError("Adjacency matrix should not contain self-loops")
    if not check_symmetric(adjacency_matrix):
        raise ValueError("Adjacency matrix should be symmetric")
    if not np.all(np.logical_or(adjacency_matrix == 0, adjacency_matrix == 1)):
        raise ValueError("Adjacency matrix should be binary")


def _approximate_from_node_membership(
    adjacencyMatrix: np.ndarray, node_memberships: np.ndarray
) -> np.ndarray:
    """Average the adjacency matrix according to the node memberships.

    Parameters
    ----------
    adjacencyMatrix : np.ndarray
        adjacency matrix of the graph
    node_memberships : np.ndarray
        node memberships

    Returns
    -------
    np.ndarray
        matrix of connection probability P
    """
    # compute the actual values of the graphon approximation
    groups = np.unique(node_memberships)
    countGroups = Counter(node_memberships)
    ngroups = len(groups)
    rho = edge_density(adjacencyMatrix)
    rho_inv = 1 / rho if rho != 0 else 1
    graphon_matrix = np.zeros((ngroups, ngroups))

    # compute the number of links between groups i and j / all possible
    # links
    for i in range(ngroups):
        for j in np.arange(i, ngroups):
            total = countGroups[groups[i]] * countGroups[groups[j]]
            graphon_matrix[i][j] = (
                np.sum(
                    adjacencyMatrix[np.where(node_memberships == groups[i])[0]][
                        :, np.where(node_memberships == groups[j])[0]
                    ]
                )
                / total
            )
            graphon_matrix[i, j] = graphon_matrix[i, j] * rho_inv
            graphon_matrix[j][i] = graphon_matrix[i, j]
    return graphon_matrix
