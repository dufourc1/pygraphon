"""Implementation of empirical degrees based estimator."""
from typing import Optional, Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_graph import _approximate_P_from_node_membership


class LG(BaseEstimator):
    """Larget Gap algorithm :cite:p:`channarond2012`.

    Parameters
    ----------
    K : int
        Number of blocks.
    """

    def __init__(self, K: int) -> None:
        if K < 1:
            raise ValueError("K must be greater than 0")
        self.K = K

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, K: Optional[int] = None
    ) -> Tuple[StepGraphon, np.ndarray]:
        if K is None:
            K = self.K
        if K > adjacency_matrix.shape[0]:
            raise ValueError("K must be smaller than the number of nodes")

        communities = self._find_communities(adjacency_matrix)
        P_hat = _approximate_P_from_node_membership(adjacency_matrix, communities)

        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat

    def _find_communities(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Find communities in the nodes based on the degree.

        Find the biggest gap in the degree distribution, and infer community as in the paper.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Adjacency matrix of the graph.

        Returns
        -------
        np.ndarray
            Node membership of each community.
        """
        num_nodes = adjacency_matrix.shape[0]

        # Compute the degree and normalized degree sequences
        degrees = np.sum(adjacency_matrix - np.diag(np.diag(adjacency_matrix)), axis=1)
        normalized_degrees = degrees / (num_nodes - 1)

        # Sort the nodes based on their normalized degree
        node_indices = np.argsort(normalized_degrees)

        # Compute the largest gaps in the normalized degree sequence
        degree_differences = np.diff(normalized_degrees)
        largest_gap_indices = np.argsort(degree_differences)[::-1][: self.K - 1]
        block_boundaries = np.concatenate(([0], np.sort(largest_gap_indices), [num_nodes]))

        # Assign nodes to blocks based on their degree sequence
        block_assignments = [
            node_indices[block_boundaries[i] + 1 : block_boundaries[i + 1]] for i in range(self.K)
        ]

        # Assign communities based on the block assignments
        communities = np.zeros(num_nodes, dtype=int)
        for i, block in enumerate(block_assignments):
            communities[block] = i

        return communities
