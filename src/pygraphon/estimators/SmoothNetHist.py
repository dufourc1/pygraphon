"""Smoothing Network Histograms Class."""

from typing import List, Optional, Tuple
from warnings import warn

import numpy as np
from sklearn.cluster import KMeans

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.assignment import Assignment
from pygraphon.estimators.networkhistogram.nethist import nethist
from pygraphon.graphons import StepGraphon
from pygraphon.utils import bic, edge_density, log_likelihood


class SmoothNetHist(BaseEstimator):
    """Implementation of the smoothing network histograms algorithm.

    From the theory of edge clustering, approximate a graphon by smoothing blocks of similar density from a histogram.

    Args:
        bandwidth (float): bandwidth of the graphon.

    Functions:
        _approximate_graphon_from_adjacency(): Estimate the graphon function from an adjacency matrix using Smoothing.
        _first_approximate_graphon_from_adjacency(): First histogram fit of the graphon function.
        _smoothing_histLC(): Smoothing the histogram using the theory of edge clustering.
        _flat_to_tensor(): Convert a list of flat vector to a tensor, each slice representing a Pij matrix.
        _from_graphon_to_adjacency(): Convert a graphon to a Pij matrix, given node assignments.
        _select_best_graphon(): Select the best graphon from a tensor of Pij matrix, using BIC.
        _log_likelihood(): Compute the log likelihood of a Pij matrix.
        _bic(): Compute the BIC of a Pij matrix.
    """

    def __init__(self, bandwidth: Optional[float] = None) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self._bic = np.inf
        self._num_par_nethist = 0
        self._num_par_smooth = np.inf

    def _approximate_graphon_from_adjacency(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        number_link_communities: Optional[int] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = False,
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Estimate the graphon function f(x,y) from an adjacency matrix.

        Parameters
        ----------
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        bandwidthHist : float
            size of the blocks (between 0 and 1), by default None

        Returns
        -------
        Tuple[StepGraphon, np.ndarray]
            approximated graphon, matrix of connection Pij of size n x n

        Raises
        ------
        ValueError
            If the number of smoothed graphon is not equal to the number of link communities.
        """
        first_fit_graphon, assignment = self._first_approximate_graphon_from_adjacency(
            adjacencyMatrix,
            bandwidthHist,
            absTol,
            maxNumIterations,
            past_non_improving,
            use_default_bandwidth,
        )
        tensor_graphon = self._smoothing_histLC(
            first_fit_graphon, adjacencyMatrix, number_link_communities
        )
        # CHECK IF THIS IS CORRECT IF NUMBER_LINK_COMMUNITIES IS SPECIFIED
        if not len(np.unique(tensor_graphon[-1].get_graphon())) == len(tensor_graphon):
            raise ValueError(
                "The number of smoothed graphon is not equal to the number of link communities"
            )
        else:
            # Convert the tensor of graphon to a tensor of Pij matrix ordered by the assignments.
            tensor_P = np.array(
                [
                    tensor_graphon[i]._get_edge_probabilities(
                        n=adjacencyMatrix.shape[0],
                        latentVarArray=assignment.labels_to_latent_variables,
                    )
                    for i in range(len(tensor_graphon))
                ]
            )
            # Select the best graphon from the tensor
            pij_matrix = self._select_best_graphon(
                tensor_P,
                adjacencyMatrix,
                number_link_communities,
            )
            return (
                StepGraphon(
                    pij_matrix,
                    bandwidthHist=first_fit_graphon.bandwidthHist,
                    initial_rho=first_fit_graphon.initial_rho,
                ),
                pij_matrix,
            )

    def _first_approximate_graphon_from_adjacency(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = False,
    ) -> Tuple[StepGraphon, Assignment]:
        """Estimate the graphon function f(x,y) from an adjacency matrix from a first histogram from [1].

        Parameters
        ----------
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
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
        bandwidthHist = h / adjacencyMatrix.shape[0]

        # ---------------------------------------------------------------------------------------------
        # Return the graphon from first fit nethist.
        graphon_nethist = StepGraphon(
            assignment_nethist.theta, bandwidthHist=bandwidthHist, initial_rho=rho
        )
        return graphon_nethist, assignment_nethist

    def _smoothing_histLC(
        self, hist_approx: StepGraphon, A: np.ndarray, number_link_communities: Optional[int] = None
    ) -> List[StepGraphon]:
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
            if number_link_communities is not None, avr_graphon is a single smoothed graphon
            with 'number_link_communities' link communities.

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
        self._num_par_nethist = nclust_hist
        if (
            number_link_communities is None
        ):  # if None, test all possible number of link communities smoothing
            avr_graphon = np.asarray([flat_graphon.copy() for _ in range(nclust_hist)])
            for i in range(nclust_hist - 1):
                if i == 0:
                    avr_graphon[i] = edge_density(A) * np.ones_like(flat_graphon)
                else:
                    kmeans = self._cluster(i + 1, flat_graphon)
                    avr_graphon[i] = self._smoothing_operator(
                        tensor_slice_graphon=avr_graphon[i],
                        kmeans=kmeans,
                        number_link_communities=i + 1,
                        flat_graphon=flat_graphon,
                        flat_areas=flat_areas,
                    )
        elif number_link_communities > nclust_hist:
            raise ValueError(
                f"The number of link communities {number_link_communities} is greater "
                + f"than the number of clusters {nclust_hist} in the histogram estimator"
            )
        elif number_link_communities == nclust_hist:
            return self._flat_to_tensor(flat_graphon, hist_approx, A.shape[0])
        else:  # else, test only the number of link communities smoothing given by the user
            # so that it has shape (1, len(flat_graphon))), for _flat_tensor to work properly
            avr_graphon = np.asarray(flat_graphon.copy()).reshape(1, -1)
            kmeans = self._cluster(number_link_communities, flat_graphon)
            avr_graphon = self._smoothing_operator(
                tensor_slice_graphon=avr_graphon[0],
                kmeans=kmeans,
                number_link_communities=number_link_communities,
                flat_graphon=flat_graphon,
                flat_areas=flat_areas,
            ).reshape(1, -1)
        return self._flat_to_tensor(avr_graphon=avr_graphon, hist_approx=hist_approx)

    def _cluster(
        self, number_link_communities: int, flat_graphon: np.ndarray, method: str = "k-means++"
    ) -> np.ndarray:
        """Clustering of the flattened graphon.

        Parameters
        ----------
        number_link_communities : int
            number of link communities
        flat_graphon : np.ndarray
            flattened graphon
        method : str, optional
            method used for the clustering, by default "k-means++"

        Returns
        -------
        np.ndarray
            labels of the clustering
        """
        kmeans = KMeans(
            n_clusters=number_link_communities, random_state=0, init=method, n_init=10
        ).fit(flat_graphon.reshape(-1, 1))

        return kmeans.labels_

    def _smoothing_operator(
        self,
        tensor_slice_graphon: np.ndarray,
        kmeans: np.ndarray,
        number_link_communities: int,
        flat_graphon: np.ndarray,
        flat_areas: np.ndarray,
    ) -> np.ndarray:
        """Smoothing operator.

        Parameters
        ----------
        tensor_slice_graphon : np.ndarray
            tensor slice containing a copy of the first graphon estimation, flattened.
        kmeans : np.ndarray
            labels of the clustering
        number_link_communities : int
            number of link communities
        flat_graphon : np.ndarray
            flattened graphon
        flat_areas : np.ndarray
            flattened areas

        Returns
        -------
        tensor_slice_graphon : np.ndarray
            tensor slice containing the smoothed graphon, smoothed with number_link_communities
        """
        for j in range(number_link_communities):
            tensor_slice_graphon[kmeans == j] = np.sum(
                flat_graphon[kmeans == j] * flat_areas[kmeans == j]
            ) / np.sum(flat_areas[kmeans == j])

        return tensor_slice_graphon

    def _flat_to_tensor(
        self, avr_graphon: np.ndarray, hist_approx: StepGraphon
    ) -> List[StepGraphon]:
        """Convert the flattened smoothed histograms to a tensor containing all possible smoothing.

        Parameters
        ----------
        avr_graphon : np.ndarray
            list of smoothed graphon in increasing order of number of link communities,
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator

        Returns
        -------
        smoothed_graphon : list of StepGraphon
            list of smoothed graphon in increasing order of number of link communities,
        """
        nrow_tile = hist_approx.graphon.shape[0]
        nclust_hist = avr_graphon.shape[0]  # number of slices in the tensor

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
        sym_KM_smooth_g = KM_smooth_g + np.transpose(KM_smooth_g, (0, 2, 1)) - diag_km
        smoothed_graphons = [
            StepGraphon(
                graphon=sym_KM_smooth_g[i],
                bandwidthHist=hist_approx.bandwidthHist,
                initial_rho=hist_approx.initial_rho,
            )
            for i in range(nclust_hist)
        ]
        return smoothed_graphons

    def _select_best_graphon(
        self,
        tensor_pij: List[np.ndarray],
        adjacencyMatrix: np.ndarray,
        number_link_communities: Optional[int] = None,
    ) -> np.ndarray:
        """Select the best graphon from the tensor of graphons by BIC.

        Parameters
        ----------
        tensor_pij : np.ndarray
            tensor of smoothed graphons, in Pij form (n x n)
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        number_link_communities : Optional[int], optional
            number of link communities, by default None

        Returns
        -------
        best_pij_matrix : np.ndarray
            best graphon matrix selected by the BIC.
        """
        if number_link_communities is None:
            k = tensor_pij.shape[0]
            bic_score = np.zeros(k)
            for i in range(k):
                num_par = len(np.unique(tensor_pij[i]))  # SHOULD BE TESTED
                log_likelihood_value = log_likelihood(tensor_pij[i], adjacencyMatrix)
                bic_score[i] = bic(log_likelihood_value, adjacencyMatrix.shape[0], num_par)
            best_index = np.argmin(bic_score)
            best_pij_matrix = tensor_pij[best_index]
            self._bic = bic_score[best_index]
            self._num_par_smooth = len(np.unique(best_pij_matrix))
        else:
            best_pij_matrix = tensor_pij
            self._bic = bic(
                log_likelihood(tensor_pij, adjacencyMatrix),
                adjacencyMatrix.shape[0],
                number_link_communities,
            )
        return best_pij_matrix

    def get_bic(self) -> float:
        """Get the BIC score of the estimated graphon.

        Returns
        -------
        float
            BIC score.
        """
        if self.fitted is False:
            warn("The model has not been fitted yet, returning np.inf.")
        return self._bic

    def get_ratio_par(self) -> float:
        """Get the ratio of the number of parameters of the smoothed graphon over the original network histogram.

        Returns
        -------
        float
            ratio of the number of parameters of smoothed vs. original
        """
        if self.fitted is False:
            warn("The model has not been fitted yet, returning np.inf.")
            return np.inf

        return self._num_par_smooth / self._num_par_nethist
