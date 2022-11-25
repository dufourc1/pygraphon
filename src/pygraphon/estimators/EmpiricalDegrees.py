"""Implementation of empirical degrees based estimator."""
from typing import Optional, Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_graph import _approximate_P_from_node_membership


class LG(BaseEstimator):
    """Larget Gap algorithm estimates graphons based on empirical degrees [1].

    Parameters
    ----------
    K : int
        Number of blocks.

    References
    ----------
    [1] Channarond, Antoine, Jean-Jacques Daudin, and Stéphane Robin. "Classification and estimation in the
    stochastic blockmodel based on the empirical degrees." Electronic Journal of Statistics 6 (2012): 2574-2601.
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
        # compute degrees and sort them
        degrees = adjacency_matrix.sum(axis=0) / (adjacency_matrix.shape[0] - 1)
        indices_sorted = np.argsort(degrees)
        degrees_sorted = degrees[indices_sorted]

        # find the K biggest differences between consecutive degrees and find corresponding indices
        ind = np.argpartition(np.diff(degrees_sorted), -self.K + 1)[-self.K + 1 :]
        gaps_indices = [0] + list(ind) + [adjacency_matrix.shape[0] - 1]

        # assign the nodes to the blocks
        blocks = []
        for i in range(self.K):
            blocks.append(indices_sorted[gaps_indices[i] : gaps_indices[i + 1]])

        communities = np.zeros(adjacency_matrix.shape[0])
        for i, block in enumerate(blocks):
            communities[block] = i
        return communities
