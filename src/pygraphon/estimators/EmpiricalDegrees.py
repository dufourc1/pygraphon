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

        communities = self._find_communities_matlab(adjacency_matrix)
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

    def _find_communities_matlab(self,G):
        blocks =  estimate_blocks_largest_gap(G, self.K)
        communities = np.zeros(G.shape[0])
        for i, block in enumerate(blocks):
            communities[block] = i
        return communities


def estimate_blocks_largest_gap(G, Q):
    '''
    Estimates the blocks of a graph using the largest gaps in the sorted normalized degree sequence.

    Args:
    - G: adjacency matrix of the graph
    - Q: number of blocks to estimate

    Returns:
    - B: list of block assignments
    '''

    if G.ndim > 2:
        G = np.sum(G, axis=2)

    n = G.shape[0]

    Deg = np.sum(G - np.diag(np.diag(G)), axis=1) # degree
    DegNorm = Deg / (n - 1) # normalized degree
    idx = np.argsort(DegNorm) # sort normalized degree

    DegNormDiff = np.diff(DegNorm) # difference
    I = np.argsort(DegNormDiff)[::-1] # find Q-1 largest gaps
    ii = np.concatenate(([0], np.sort(I[0:Q-1]), [n])) # ii = Q-1 indices of largest gaps

    B = [None] * Q
    for q in range(Q):
        B[q] = idx[ii[q]+1:ii[q+1]]

    return B
