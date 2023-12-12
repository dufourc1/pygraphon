"""Random utils to deal with graphs."""
from collections import Counter

import networkx as nx
import numba as nb
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


def _approximate_P_from_node_membership(
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
        edge connectivity
    """
    # compute the actual values of the graphon approximation
    groups = np.unique(node_memberships)
    countGroups = Counter(node_memberships)
    P = np.zeros_like(adjacencyMatrix)
    node_memberships = node_memberships.astype(int)
    ngroups = max(node_memberships) + 1
    values = np.zeros((ngroups, ngroups))
    # compute the number of links between groups i and j / all possible
    # links
    for i in range(ngroups):
        for j in np.arange(i, ngroups):
            total = (
                countGroups[groups[i]] * countGroups[groups[j]]
                if i != j
                else countGroups[groups[i]] * (countGroups[groups[j]] - 1)
            )
            values[i, j] = compute_edge_between_groups(
                adjacencyMatrix,
                np.where(node_memberships == groups[i])[0],
                np.where(node_memberships == groups[j])[0],
            )
            values[i, j] /= total
            values[j, i] = values[i, j]

    for i in range(P.shape[0]):
        for j in range(i + 1, P.shape[1]):
            P[i, j] = values[node_memberships[i], node_memberships[j]]
            P[j, i] = P[i, j]
    return P


@nb.jit(nopython=True)
def compute_edge_between_groups(A, indices_1, indices_2):
    """Compute the number of edges between two groups of nodes.

    If the two groups are the same, the number of edges will be double counted.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph
    indices_1 : np.ndarray
        Indices of the first group
    indices_2 : np.ndarray
        Indices of the second group

    Returns
    -------
    int
        Number of edges between the two groups
    """
    result = 0
    for i in indices_1:
        for j in indices_2:
            result += A[i, j]
    return result
