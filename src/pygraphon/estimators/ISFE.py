import numpy as np
from typing import Optional, Tuple, List
from pygraphon.graphons import StepGraphon
from pygraphon.estimators.BaseEstimator import BaseEstimator

from pygraphon.estimators import HistogramEstimator


class ISFE(BaseEstimator):
    """Iterative step-function estimator :cite:p:`cai2014`."""

    def __init__(
        self,
        number_iterations: int = 2,
        decay: float = 0.1,
        epsilon: float = 1e-4,
        min_number_classes: int = 2,
    ) -> None:
        super().__init__()
        self.number_iterations = number_iterations
        self.decay = decay
        self.epsilon = epsilon
        self.min_number_classes = min_number_classes

    def _approximate_graphon_from_adjacency(
        self,
        adjacency_matrix: np.ndarray,
        initial_partition: Optional[List[List[int]]] = None,
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Estimate the graphon function f(x,y) from an adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjancency matrix of the graph
        initial_partition : Optional[List[List[int]]], optional
            Initial partition of the nodes to estimate the step function.
            By default :obj:`None`  will use a random partition into
            `self.min_number_classes` classes.

        Returns
        -------
        Tuple[StepGraphon, np.ndarray]
            approximated graphon and matrix of connection Pij of size n x n
        """
        # get the initial partition
        partition = self.get_initial_partition(
            adjacency_matrix, initial_partition
        )
        # update the partition iteratively
        for _ in range(self.number_iterations):
            partition = self.update_partition(adjacency_matrix, partition)

        # convert partition to node labels
        node_labels = self._partition_to_node_membership(partition)
        # given the partition, we can estimate the graphon
        graphon_matrix = (
            HistogramEstimator._approximate_graphon_from_adjacency(
                adjacency_matrix, node_labels
            )
        )

        # build the edge probability matrix
        edge_probability_matrix = np.zeros(
            (len(node_labels), len(node_labels))
        )
        for i in range(edge_probability_matrix.shape[0]):
            for j in np.arange(i + 1, edge_probability_matrix.shape[0]):
                edge_probability_matrix[i, j] = graphon_matrix[
                    int(node_labels[i]), int(node_labels[j])
                ]
                edge_probability_matrix[j, i] = edge_probability_matrix[i, j]

        return graphon_matrix, edge_probability_matrix

    def get_initial_partition(
        self,
        adjacency_matrix: np.ndarray,
        initial_partition: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """Get the initial partition of the graph."""
        if initial_partition is None:
            initial_labels = np.random.randint(
                0, self.min_number_classes, size=adjacency_matrix.shape[0]
            )
            initial_partition = self._node_membership_to_partition(
                initial_labels
            )
        return initial_partition

    def update_partition(
        self, adjacency_matrix: np.ndarray, partition: List[List[int]]
    ) -> List[List[int]]:
        """Update the partition according to the ISFE algorithm."""
        raise NotImplementedError(
            "algorithm unclear from paper and no implementation available"
        )
        size_partition = [len(c) for c in partition]
        new_partition = [
            [0],
        ]
        n_nodes = adjacency_matrix.shape[0]
        weighted_density = np.zeros(len(partition))
        for i in range(1, n_nodes):
            for j in range(len(partition)):
                weighted_density[j] = (
                    adjacency_matrix[i, partition[j]].sum()
                    / size_partition[j]
                )

        return new_partition

    def _partition_to_node_membership(
        self, partition: List[List[int]]
    ) -> np.ndarray:
        n_nodes = sum(len(c) for c in partition)
        node_membership = np.zeros(n_nodes, dtype=int)
        for i, c in enumerate(partition):
            node_membership[c] = i
        return node_membership

    def _node_membership_to_partition(
        self, node_membership: np.ndarray
    ) -> List[List[int]]:
        partition = []
        for i in range(node_membership.max() + 1):
            partition.append(np.where(node_membership == i)[0])
        return partition
