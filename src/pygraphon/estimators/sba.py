"""Implementation of the SBA estimator."""
import random
from typing import List, Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_graph import _approximate_P_from_node_membership


class SBA(BaseEstimator):
    """Estimate graphon base on SB approximation.

    Parameters
    ----------
    delta : float, optional
        precision parameter larger than 0, by default 0.2
    """

    def __init__(self, delta: float = 0.2) -> None:
        if delta < 0:
            raise ValueError("delta must be larger than 0")
        self.delta = delta

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:

        n = adjacency_matrix.shape[0]
        k = 0
        set_index = set(range(n))
        membership = np.zeros(n, dtype=np.int64)

        # iterate over all nodes
        while set_index:
            # set membership of the pivot
            index = random.sample(set_index, 1)[0]
            set_index.remove(index)
            membership[index] = k

            # list all of the nodes non attributed to a community
            non_clustered_indices = np.array(list(set_index))

            # compute the distance between the pivot and the other nodes
            distances = self._compute_dij(
                index_i=index,
                index_j=list(non_clustered_indices),
                adjacency_matrix=adjacency_matrix,
            )

            # find the close nodes and add them to the community
            close_j = non_clustered_indices[np.where(distances < self.delta**2)[0]]
            for j in close_j:
                membership[j] = k
                set_index.remove(j)
            # update community number
            k += 1

        # format output and return
        P_hat = _approximate_P_from_node_membership(adjacency_matrix, membership)
        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat

    def _compute_dij(
        self, index_i: int, index_j: List[int], adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute the distance betweem two nodes as defined in [1] eq. 5.

        Parameters
        ----------
        index_i : int
            index of the reference node
        index_j : np.ndarray
            list of indices of the other nodes to compute the distance
        adjacency_matrix : np.ndarray
            adjacency matrix of the graph

        Returns
        -------
        np.ndarray
            distance (len(index_j),)

        Notes
        ------
        In the case of a unique simple undirected graph, the formula boils down to: d_ij = (c__ii - 2*c_ij + c_jj)
        which is what is implemented here. This is not the case if multiple inputs are provided.
        """
        n = adjacency_matrix.shape[0]
        c = np.dot(adjacency_matrix, adjacency_matrix)[index_i, :]
        return c[index_i] / (n - 1) - 2 * c[index_j] / (n - 2) + c[index_j] / (n - 1)
