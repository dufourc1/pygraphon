"""Smoothing Network Histograms Class."""
from __future__ import annotations

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
        bandwidth (float): bandwidth of the original non-smoothed :py:class:`~pygraphon.estimators.HistogramEstimator`
        graphon.

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
        self._num_par_nethist = -1
        self._num_par_smooth = -1
        self._bic_nethist = np.inf

    def _approximate_graphon_from_adjacency(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        number_link_communities: Optional[int] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = True,
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
        """
        first_fit_graphon, assignment = self._first_approximate_graphon_from_adjacency(
            adjacencyMatrix,
            bandwidthHist,
            absTol,
            maxNumIterations,
            past_non_improving,
            use_default_bandwidth,
        )

        tensor_graphon, n_link_com = self._smoothing_histLC(
            first_fit_graphon, adjacencyMatrix, number_link_communities
        )

        return self._select_best_graphon(
            tensor_graphon,
            adjacencyMatrix,
            assignment=assignment,
            n_link_com=n_link_com,
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
            bandwidthHist = self.bandwidth

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
        self._bic_nethist = bic(
            log_likelihood_val=assignment_nethist.log_likelihood,
            n=adjacencyMatrix.shape[0],
            num_par=len(np.unique(assignment_nethist.theta)),
        )

        # ---------------------------------------------------------------------------------------------
        # Return the graphon from first fit nethist.
        graphon_nethist = StepGraphon(
            assignment_nethist.theta,
            bandwidthHist=bandwidthHist,
            initial_rho=rho,
        )
        return graphon_nethist, assignment_nethist

    def _smoothing_histLC(
        self,
        hist_approx: StepGraphon,
        A: np.ndarray,
        number_link_communities: Optional[int] = None,
    ) -> Tuple[List[StepGraphon], List[int]]:
        """Smoothing of the histogram estimator using k-means++ [1].

        Parameters
        ---------
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator
        A : np.ndarray
            adjancency matrix of the graph
        number_link_communities : int, optional
            number of link communities to be obtained after smoothing, by default None.
            will try all of them.

        Returns
        -------
        Tuple[List[StepGraphon], List[int]]
            py:obj:`(avr_graphon, n_link_com)`
            list of kmean-smoothed graphon in increasing order of number of link communities,
            avr_graphon[i] is the smoothed graphon with i+1 link communities.
            if number_link_communities is not None, avr_graphon is a single smoothed graphon
            with 'n_link_com' link communities.

        Raises
        ------
        ValueError
            If the number of link communities prompted is greater than the max number of link communities.

        References
        ----------
        [1] Arthur, D. and Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding.
        """
        # Flattening the graphon and the areas
        nrow_tile = hist_approx.graphon.shape[0]
        flat_graphon = hist_approx.graphon[np.triu_indices(nrow_tile)]
        flat_areas = hist_approx.areas[np.triu_indices(nrow_tile)]

        # Number of link communities and input check
        self._num_par_nethist = flat_graphon.size
        if number_link_communities is not None and number_link_communities > self._num_par_nethist:
            raise ValueError(
                f"The number of link communities {number_link_communities} is greater "
                + f"than the number of clusters {self._num_par_nethist} in the histogram estimator"
            )

        clusters = (
            list(range(1, self._num_par_nethist + 1))
            if number_link_communities is None
            else [number_link_communities]
        )
        avr_graphon = np.asarray([flat_graphon.copy() for _ in clusters])

        # smoothing based on kmeans if needed
        for index, number_groups in enumerate(clusters):
            # erdos renyi approximation
            if number_groups == 1:
                avr_graphon[index] = edge_density(A) * np.ones_like(flat_graphon)
            # original histogram estimator
            elif number_groups == self._num_par_nethist:
                pass
            else:
                avr_graphon[index] = self._smoothing_operator(
                    tensor_slice_graphon=avr_graphon[index],
                    labels=self._cluster(number_groups, flat_graphon),
                    number_link_communities=number_groups,
                    flat_graphon=flat_graphon,
                    flat_areas=flat_areas,
                )

        return (
            self._flat_to_tensor(avr_graphon=avr_graphon, hist_approx=hist_approx),
            clusters,
        )

    def _cluster(
        self,
        number_link_communities: int,
        flat_graphon: np.ndarray,
        method: Optional[str] = "k-means++",
    ) -> np.ndarray:
        """Clustering of the flattened graphon.

        Parameters
        ----------
        number_link_communities : int
            number of link communities
        flat_graphon : np.ndarray
            flattened graphon
        method : str.
            method used for the clustering, by default "k-means++"

        Returns
        -------
        np.ndarray
            labels of the clustering
        """
        kmeans = KMeans(
            n_clusters=number_link_communities,
            random_state=0,
            init=method,
            n_init=10,
        ).fit(flat_graphon.reshape(-1, 1))

        return kmeans.labels_

    def _smoothing_operator(
        self,
        tensor_slice_graphon: np.ndarray,
        labels: np.ndarray,
        number_link_communities: int,
        flat_graphon: np.ndarray,
        flat_areas: np.ndarray,
    ) -> np.ndarray:
        """Merge the graphon values based on the edge communities clustering.

        Update the value inside a cluster by taking the average of the graphon values
        weighted by the areas.

        Parameters
        ----------
        tensor_slice_graphon : np.ndarray
            tensor slice containing a copy of the first graphon estimation, flattened.
        labels : np.ndarray
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
            labels_j = np.argwhere(labels == j).reshape(-1)
            tensor_slice_graphon[labels_j] = np.sum(
                flat_graphon[labels_j] * flat_areas[labels_j]
            ) / np.sum(flat_areas[labels_j])

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
                initial_rho=None,
            )
            for i in range(nclust_hist)
        ]
        return smoothed_graphons

    def _select_best_graphon(
        self,
        tensor_graphon: List[StepGraphon],
        adjacencyMatrix: np.ndarray,
        assignment: Assignment,
        n_link_com: List[int],
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Select the best graphon from the tensor of graphons by BIC.

        Parameters
        ----------
        tensor_graphon : List[StepGraphon]
            tensor of smoothed graphons of shape (n_graphons, n_block, n_block),
            where n_block is the number of blocks in the original fit graphon.
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        assignment : Assignment
            assignment of the graph of the graph

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            best graphon matrix and P_ij selected by the BIC.
        """
        latentVarArray = assignment.labels_to_latent_variables()
        n_nodes = adjacencyMatrix.shape[0]

        if len(tensor_graphon) == 1:
            best_graphon = tensor_graphon[0]
            best_pij = best_graphon._get_edge_probabilities(
                n=n_nodes, latentVarArray=latentVarArray
            )
            best_bic = bic(
                log_likelihood(best_pij, adjacencyMatrix),
                n=n_nodes,
                num_par=n_link_com[0],
            )
            self._num_par_smooth = n_link_com
        else:
            best_bic = np.inf
            best_graphon = tensor_graphon[0]
            best_pij = np.ones((n_nodes, n_nodes)) / 2
            for i, graphon in enumerate(tensor_graphon):
                pij = graphon._get_edge_probabilities(n=n_nodes, latentVarArray=latentVarArray)
                log_likelihood_value = log_likelihood(pij, adjacencyMatrix)
                bic_candidate = bic(log_likelihood_value, n=n_nodes, num_par=n_link_com[i])
                if bic_candidate < best_bic:
                    best_bic = bic_candidate
                    best_graphon = graphon
                    best_pij = pij
                    self._num_par_smooth = n_link_com[i]

        self._bic = best_bic
        return best_graphon, best_pij

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
