from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


class Dist(ABC):
    @abstractmethod
    def calculate_distance(self, adjacency_matrix):
        """
        Calculate the distance matrix from the given adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            The adjacency matrix of the graph.

        Returns
        -------
        np.ndarray
            The distance matrix.
        """
        pass


class SimpleDifferenceDistance(Dist):
    """Simple difference distance between the degree distributions of the nodes."""

    def calculate_distance(self, adjacency_matrix):
        m = adjacency_matrix.shape[0]
        distance_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                distance_matrix[i, j] = np.sum(np.abs(adjacency_matrix[i] - adjacency_matrix[j]))
        return distance_matrix


class JensenShannonDivergenceDistance(Dist):
    """Jensen-Shannon divergence distance between the degree distributions of the nodes."""

    def calculate_distance(self, adjacency_matrix):
        return compute_distance_matrix(
            adjacency_matrix, lambda p, q: jensenshannon(p, q) ** 2
        )  # Squared to match the usual output scale


class EarthMoversDistance(Dist):
    """Earth Movers Distance between the degree distributions of the nodes."""

    def calculate_distance(self, adjacency_matrix):
        return compute_distance_matrix(adjacency_matrix, wasserstein_distance)


class HellingerDistance(Dist):
    """Hellinger distance between the degree distributions of the nodes."""

    def calculate_distance(self, adjacency_matrix):
        return compute_distance_matrix(
            adjacency_matrix, lambda p, q: np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
        )


class BhattacharyyaDistance(Dist):
    """Bhattacharyya distance between the degree distributions of the nodes."""

    def calculate_distance(self, adjacency_matrix):
        return compute_distance_matrix(
            adjacency_matrix, lambda p, q: -np.log(np.sum(np.sqrt(p * q)))
        )


def compute_distance_matrix(adjacency_matrix: np.ndarray, distance_func: callable) -> np.ndarray:
    """Calculate the distance matrix based on comparing the degree distributions of each node.
    By degree distribution, we mean the distribution of the degrees of the neighbors of a node.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix of the graph.
    distance_func : callable
        The distance function to use to compare the degree distributions of the nodes.
        The function should take two 1D arrays as input and return a scalar distance value.

    Returns
    -------
    np.ndarray
        The distance matrix.
    """
    m = adjacency_matrix.shape[0]
    distance_matrix = np.zeros((m, m))
    degrees = adjacency_matrix.sum(axis=0, dtype=int)
    max_degree = np.max(degrees)

    # Use element-wise multiplication to replace each 1 in the adjacency matrix with the degree of the connected node
    adjacent_degrees_matrix = adjacency_matrix * degrees

    # Calculate the degree distributions for each node
    degree_distributions = [
        np.histogram(adjacent_degrees_matrix[i], bins=np.arange(max_degree + 2), density=True)[0]
        for i in range(m)
    ]

    # Calculate the distribution of degrees of adjacent nodes for each node
    for i in range(m):
        for j in range(i + 1, m):
            distance = distance_func(degree_distributions[i], degree_distributions[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance  # Symmetric

    return distance_matrix
