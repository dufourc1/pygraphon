"""Smoothing Network Histograms Class"""

from typing import Optional, Tuple

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.HistogramEstimator import HistogramEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_graph import edge_density
from pygraphon.estimators.networkhistogram.nethist import nethist

import numpy as np

from sklearn.cluster import KMeans


class SmoothNetHist(BaseEstimator):

    """Implementation of the smoothing network histograms algorithm.
    From the theory of edge clustering, approximate a graphon by smoothing blocks of similar density from a histogram.

    Args:

        bandwidth (float): bandwidth of the graphon.

    """

    def __init__(self, bandwidth: Optional[float] = None) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def _approximate_graphon_from_adjacency(self,
                                            adjacencyMatrix: np.ndarray,
                                            bandwidthHist: Optional[float] = None,
                                            number_link_communities: Optional[int] = None,
                                            absTol: float = 2.5 * 1e-4,
                                            maxNumIterations: int = 500,
                                            past_non_improving: int = 3,
                                            use_default_bandwidth: bool = False,
                                            ) -> Tuple[StepGraphon, np.ndarray, float]:
        """Estimate the graphon function f(x,y) from an adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjacency matrix of the graph
        bandwidthHist : float
            size of the blocks (between 0 and 1), by default None

        Returns
        -------
        Tuple[StepGraphon, np.ndarray]
            approximated graphon, matrix of connection Pij of size n x n, and bandwidth of the original histogram.
        """

        first_fit_graphon, assignment = self._first_approximate_graphon_from_adjacency(
            adjacencyMatrix, bandwidthHist, absTol, maxNumIterations, past_non_improving, use_default_bandwidth)
        tensor_graphon = self._smoothing_histLC(
            first_fit_graphon, adjacencyMatrix, number_link_communities)
        # Convert the tensor of graphon to a tensor of Pij matrix ordered by the assignments.
        tensor_P = np.array(
            [self._from_graphon_to_adjacency(
                tensor_graphon[i], assignment) for i in tensor_graphon.shape[0]]
        )
        # Select the best graphon from the tensor
        graphon_matrix = self._select_best_graphon(
            tensor_P, adjacencyMatrix, number_link_communities)
        return StepGraphon(graphon_matrix, bandwidthHist=first_fit_graphon.bandwidthHist, initial_rho= edge_density(adjacencyMatrix)) , graphon_matrix, first_fit_graphon.bandwidthHist

    def _first_approximate_graphon_from_adjacency(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = False,
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Estimate the graphon function f(x,y) from an adjacency matrix from a first histogram from [1].

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjancency matrix of the graph
        bandwidthHist : float
            size of the blocks (between 0 and 1), by default None

        Returns
        -------
        Tuple[StepGraphon, np.ndarray]
            first approximated graphon, matrix of connection Pij of size n x n.

        References
        ----------
            [1]: 2013 Sofia C. Olhede and Patrick J. Wolfe (arXiv:1312.5306)
        """
        rho = edge_density(adjacencyMatrix)
        # ---------------------------------------------------------------------------------------------
        # This section is the same HistogramEstimator.py _approximate
        # network histogram approximation
        if bandwidthHist is None and use_default_bandwidth:
            bandwidthHist = self.bandwidthHist

        if bandwidthHist is None:
            h = None
        else:
            h = int(bandwidthHist * adjacencyMatrix.shape[0])
        assignment_nethist, h = nethist(
            A=adjacencyMatrix,
            h=h,
            absTol=absTol,
            maxNumIterations=maxNumIterations,
            past_non_improving=past_non_improving,
        )
        # ---------------------------------------------------------------------------------------------
        # Return the graphon from first fit nethist.
        graphon_nethist = StepGraphon(
            assignment_nethist.theta, bandwidthHist=h, initial_rho=rho)
        return graphon_nethist, assignment_nethist

    def _smoothing_histLC(self,
                          hist_approx: StepGraphon,
                          A: np.ndarray,
                          number_link_communities: Optional[int] = None
                          ):
        """Smoothing of the histogram estimator using k-means++ [1].

        Parameters
        ---------
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator
        A : np.ndarray
            adjancency matrix of the graph
        number_link_communities : int, optional
            number of link communities to be obtained after smoothing, by default None

        Returns
        -------
        avr_graphon : np.ndarray
            list of kmean-smoothed graphon in increasing order of number of link communities,
            avr_graphon[i] is the smoothed graphon with i+1 link communities.
            if number_link_communities is not None, avr_graphon is a single smoothed graphon with 'number_link_communities' link communities.

        References
        ----------
        [1] Arthur, D. and Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding.
        """
        # Flattening the graphon and the areas
        nrow_tile = hist_approx.graphon.shape[0]
        flat_graphon = hist_approx.graphon[np.triu_indices(nrow_tile)]
        flat_areas = hist_approx.areas[np.triu_indices(nrow_tile)]
        
        # Clustering according to kmeans
        nclust_hist = len(np.unique(flat_graphon))
        if number_link_communities is None:  # if None, test all possible number of link communities smoothing
            avr_graphon = np.asarray([flat_graphon.copy()
                                     for _ in range(nclust_hist)])
            for i in range(nclust_hist - 1):
                if i == 0:
                    avr_graphon[i] = edge_density(
                        A) * np.ones_like(flat_graphon)
                else:
                    kmeans = KMeans(n_clusters=i + 1, random_state=0, init="k-means++", n_init=10).fit(
                        flat_graphon.reshape(-1, 1)
                    )
                    for j in range(i + 1):  # average tiles of same community
                        avr_graphon[i][kmeans.labels_ ==
                                       j] = np.sum(flat_graphon[kmeans.labels_ == j]*flat_areas[kmeans.labels_ == j])/np.sum(flat_areas[kmeans.labels_ == j])

        else:  # else, test only the number of link communities smoothing given by the user
            avr_graphon = np.asarray(flat_graphon.copy())
            kmeans = KMeans(n_clusters=number_link_communities, random_state=0, init="k-means++", n_init=10).fit(
                flat_graphon.reshape(-1, 1)
            )
            for j in range(number_link_communities):
                avr_graphon[kmeans.labels_ ==
                            j] = np.sum(flat_graphon[kmeans.labels_ == j]*flat_areas[kmeans.labels_ == j])/np.sum(flat_areas[kmeans.labels_ == j])
                
        return self._flat_to_tensor(avr_graphon, hist_approx, A.shape[0])

    def _flat_to_tensor(self,
                        avr_graphon: np.ndarray,
                        hist_approx: StepGraphon,
                        n: int
                        ):
        """Convert the flattened smoothed histograms to a tensor containing all possible smoothing.

        Parameters
        ----------
        avr_graphon : np.ndarray
            list of smoothed graphon in increasing order of number of link communities,
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator
        n : int
            number of nodes of the graph

        Returns
        -------
        P_smooth : np.ndarray
            matrix of connection Pij of size n x n for each smoothed graphon
        """

        nrow_tile = hist_approx.graphon.shape[0]
        nclust_hist = avr_graphon.shape[0]

        # deflatte the smoothed histograms and put them in a tensor
        KM_smooth_g = np.zeros((nclust_hist, nrow_tile, nrow_tile))
        i, j = np.triu_indices(nrow_tile)
        KM_smooth_g[:, i, j] = avr_graphon
        # Make it symmetric
        i, j = np.diag_indices(nrow_tile)
        diag_km_vec = np.diagonal(KM_smooth_g, axis1=1, axis2=2)
        diag_km = np.zeros((nclust_hist, nrow_tile, nrow_tile))
        diag_km[:, i, j] = diag_km_vec
        # X+X^T-diag(X) on axis (1,2) of tensor
        sym_KM_smooth_g = KM_smooth_g + \
            np.transpose(KM_smooth_g, (0, 2, 1)) - diag_km
        smoothed_graphons = [
            StepGraphon(
                graphon=sym_KM_smooth_g[i],
                bandwidthHist=hist_approx.bandwidthHist,
                initial_rho=hist_approx.initial_rho,
            )
            for i in range(nclust_hist)
        ]
        return smoothed_graphons

    def _from_graphon_to_adjacency(
            self,
            graphon_matrix: np.ndarray,
            assignment_nethist: np.ndarray,
    ) -> np.ndarray:
        """From a graphon matrix to an adjacency matrix.

        Parameters
        ----------
        graphon_matrix : np.ndarray
            graphon matrix
        assignment_nethist : np.ndarray
            assignment of the histogram estimator
        n : int
            number of nodes

        Returns
        -------
        edge_probability_matrix : np.ndarray
            matrix of connection Pij of size n x n"""
        
        groupmembership = assignment_nethist.labels

        # fills in the edge probability matrix from the value of the graphon
        edge_probability_matrix = np.zeros(
            (len(groupmembership), len(groupmembership)))
        for i in range(edge_probability_matrix.shape[0]):
            for j in np.arange(i + 1, edge_probability_matrix.shape[0]):
                edge_probability_matrix[i, j] = graphon_matrix[
                    int(groupmembership[i]), int(groupmembership[j])
                ]
                edge_probability_matrix[j, i] = edge_probability_matrix[i, j]
        return edge_probability_matrix

    def _select_best_graphon(
            self,
            tensor_graphon: np.ndarray,
            adjacencyMatrix: np.ndarray,
            number_link_communities: Optional[int] = None,
            ) -> np.ndarray:
        """Select the best graphon from the tensor of graphons by BIC.

        Parameters
        ----------
        tensor_graphon : np.ndarray
            tensor of smoothed graphons
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        number_link_communities : Optional[int], optional
            number of link communities, by default None
        
        Returns
        -------
        best_graphon_matrix : np.ndarray
            best graphon matrix selected by the BIC.
        """
        if number_link_communities is None:
            k = tensor_graphon.shape[0]
            bic_score = np.zeros(k)
            for i in range(k):
                num_par = len(np.unique(tensor_graphon[i])) #SHOULD BE TESTED
                log_likelihood_value = log_likelihood(tensor_graphon[i], adjacencyMatrix)
                bic_score[i] = bic(log_likelihood_value, adjacencyMatrix.shape[0], num_par)
            best_graphon_matrix = tensor_graphon[np.argmin(bic_score)]
        else:
            best_graphon_matrix = tensor_graphon
        return best_graphon_matrix
    

def log_likelihood(probs: np.ndarray, A: np.ndarray, eps:float = 1e-10) -> float:
    """Compute the log-likelihood of the graphon given the adjacency matrix.

    Suppose probs and A are aligned.

    Parameters
    ----------
    probs : np.ndarray
        edge probability matrix
    A : np.ndarray
        adjacency matrix of the graph
    eps : float
        small value to avoid numerical issues with log(0)

    Returns
    -------
    float
        log-likelihood of the graphon given the adjacency matrix
    """
    probs_stable = np.clip(probs, eps, 1 - eps)
    return np.sum(A * np.log(probs_stable) + (1 - A) * np.log(1 - probs_stable))

def bic(log_likelihood_val: float, n: int, num_par: int) -> float:
    """Compute the BIC score of the graphon.

    Parameters
    ----------
    log_likelihood_val : float
        log-likelihood of the graphon given the adjacency matrix
    n : int
        number of nodes of the graph
    num_par : int
        number of parameters of the graphon

    Returns
    -------
    float
        BIC score of the graphon
    """
    return -2 * log_likelihood_val + num_par * np.log(n*(n-1)/2)