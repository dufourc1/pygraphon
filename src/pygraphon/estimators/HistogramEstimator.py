"""Network histogram class."""
from collections import Counter
from typing import Tuple

import numpy as np

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.utils.nethist import nethist
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_graph import edge_density


class HistogramEstimator(BaseEstimator):
    """Implements the histogram estimator from Universality of block model approximation [1].

    Approximate a graphon from a single adjacency matrix. Size of blocks can be  determined automaticaly.

    Args:
        bandwithHist (float, optional): size of the block of the histogram. If None, automatically derived
            from the observation. Defaults to None.


    Sources:
        [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
    """

    def __init__(self, bandwithHist: float = None) -> None:

        super().__init__()
        self.bandwidthHist = bandwithHist

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, bandwidthHist: float = None
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Estimate the graphon function f(x,y) from an adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjancency matrix of the graph
        bandwidthHist : float
            size of the blocks (between 0 and 1), by default None

        Returns
        -------
        Tuple[StepGraphon, np.ndarray]
            approximated graphon and matrix of connection Pij of size n x n
        """
        rho = edge_density(adjacency_matrix)
        if bandwidthHist is None:
            bandwidthHist = self.bandwidthHist
        graphon_matrix, P, h = self._approximate(adjacency_matrix, bandwidthHist)
        return StepGraphon(graphon_matrix, rho, h), P

    def _approximate(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: float = None,
        use_default_bandwidth: bool = False,
        return_membership: bool = False,
    ) -> Tuple[Tuple[np.ndarray], float]:
        """Use function from Universality of block model approximation [1].

        Approximate a graphon from a single adjacency matrix.

        Parameters
        ----------
        adjacencyMatrix : np.ndarray
            adjancency matrix of the graph
        bandwidthHist : float
            size of the block of the histogram. Defaults to None
        use_default_bandwidth : bool
            if True, use the default bandwidth. Defaults to False.
        return_membership : bool
            if true return also the node membership, by default False

        Returns
        -------
        Tuple[Tuple[np.ndarray], float]
            graphon_matrix, edge_probability_matrix, h. graphon_matrix is the block model
            graphon and P is the edge probability matrix
            corresponding to the adjacency matrix. h is the size of the block

        Sources
        -------
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
        """
        # network histogram approximation
        if bandwidthHist is None and use_default_bandwidth:
            bandwidthHist = self.bandwidthHist

        if bandwidthHist is None:
            h = None
        else:
            h = int(bandwidthHist * adjacencyMatrix.shape[0])
        groupmembership, h, _ = nethist(A=adjacencyMatrix, h=h)

        if bandwidthHist is None:
            bandwidthHist = h / adjacencyMatrix.shape[0]

        graphon_matrix = self._approximate_from_node_membership(adjacencyMatrix, groupmembership)

        # fills in the edge probability matrix from the value of the graphon
        edge_probability_matrix = np.zeros((len(groupmembership), len(groupmembership)))
        for i in range(edge_probability_matrix.shape[0]):
            for j in np.arange(i + 1, edge_probability_matrix.shape[0]):
                edge_probability_matrix[i, j] = graphon_matrix[
                    int(groupmembership[i]), int(groupmembership[j])
                ]
                edge_probability_matrix[j, i] = edge_probability_matrix[i, j]
        if return_membership:
            return graphon_matrix, edge_probability_matrix, bandwidthHist, groupmembership
        return graphon_matrix, edge_probability_matrix, bandwidthHist

    @staticmethod
    def _approximate_from_node_membership(
        adjacencyMatrix: np.ndarray, node_memberships: np.ndarray
    ) -> np.ndarray:
        # compute the actual values of the graphon approximation
        groups = np.unique(node_memberships)
        countGroups = Counter(node_memberships)
        ngroups = len(groups)
        # rho = edge_density(adjacencyMatrix)
        # rho_inv = 1 / rho if rho != 0 else 0
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
                graphon_matrix[i, j] = graphon_matrix[i, j]  # * rho_inv
                graphon_matrix[j][i] = graphon_matrix[i, j]
        return graphon_matrix
