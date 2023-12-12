"""Smoothing Network Histograms Class."""
from __future__ import annotations

from typing import List, Optional, Tuple
from warnings import warn

import numpy as np
from loguru import logger
from scipy.special import comb
from sklearn.cluster import KMeans

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.assignment import Assignment
from pygraphon.estimators.networkhistogram.nethist import nethist
from pygraphon.graphons import StepGraphon
from pygraphon.utils import (
    aic,
    bic,
    edge_density,
    elbow_point,
    fpe,
    hqic,
    log_likelihood,
    mallows_cp,
)

CRITERIONS = {
    "bic": bic,
    "aic": aic,
    "elbow": lambda norm, *args, **kwargs: norm,
    "mallows_cp": mallows_cp,
    "hqic": hqic,
    "fpe": fpe,
}


class SmoothNetHist(BaseEstimator):
    """Implementation of the smoothing network histograms algorithm.

    From the theory of edge clustering, approximate a graphon by smoothing blocks of similar density from a histogram.

    Args:
        criterion (str): criterion used for the model selection, either 'aic', 'bic',
        'elbow', 'mallows_cp', 'hqic', 'fpe'.
        bandwidth (float): bandwidth of the original non-smoothed :py:class:`~pygraphon.estimators.HistogramEstimator`
        graphon.
    """

    def __init__(self, criterion: str = "bic", bandwidth: Optional[float] = None) -> None:
        super().__init__()
        if criterion.lower() not in CRITERIONS.keys() and criterion.lower() != "all":
            raise ValueError(
                f"Criterion {criterion} not available, please choose between {list(CRITERIONS.keys())}"
            )
        elif criterion.lower() == "all":
            self.gof = None
            self.criterion_name = "all"
            self.criterion_values = {k: np.inf for k in CRITERIONS.keys()}
            self.best_pairs = {k: {"graphon": None, "pij": None} for k in CRITERIONS.keys()}
            self._criterion_nethist = {k: np.inf for k in CRITERIONS.keys()}
            self._num_par_smooth = {k: -1 for k in CRITERIONS.keys()}
        else:
            # self.gof is the criterion function
            self.gof = CRITERIONS[criterion.lower()]
            # self.criterion_name is the name of the criterion
            self.criterion_name = criterion.lower()
            self.criterion_values = {self.criterion_name: np.inf}
            self.best_pairs = {self.criterion_name: {"graphon": None, "pij": None}}
            self._criterion_nethist = {self.criterion_name: np.inf}
            self._num_par_smooth = {self.criterion_name: -1}
        self.kmeans_labels_evolution = {"labels": [], "nLC": []}
        self._num_par_nethist = -1
        self.bandwidth = bandwidth
        self._tolerance = 1e-4
        logger.warning("toleration is set to 1e-4, arbitrary value, argue")
        self._first_fit_graphon = None
        self.cluster_it_list = []
        self.all_cluster_graphons = None

    def _approximate_graphon_from_adjacency(
        self,
        adjacencyMatrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        number_link_communities: Optional[int] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = True,
        progress_bar: bool = False,
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
            progress_bar,
        )
        self._first_fit_graphon = first_fit_graphon

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
        progress_bar: bool = False,
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
            progress_bar=progress_bar,
        )

        bandwidthHist = h / adjacencyMatrix.shape[0]
        # ---------------------------------------------------------------------------------------------
        # Return the graphon from first fit nethist.
        graphon_nethist = StepGraphon(
            assignment_nethist.theta,
            bandwidthHist=bandwidthHist,
        )
        # Update criterion
        pij = graphon_nethist._get_edge_probabilities(
            n=adjacencyMatrix.shape[0],
            latentVarArray=assignment_nethist.labels_to_latent_variables(),
        )
        params = dict(
            log_likelihood_val=assignment_nethist.log_likelihood,
            n=adjacencyMatrix.shape[0],
            num_par=comb(assignment_nethist.theta.shape[0] + 1, 2),
            norm=np.linalg.norm(pij - adjacencyMatrix)
            / (adjacencyMatrix.shape[0] ** 2 - adjacencyMatrix.shape[0]),
            var=np.var(pij - adjacencyMatrix, ddof=adjacencyMatrix.shape[0]),
        )
        list_criterion = (
            [self.criterion_name] if self.criterion_name != "all" else CRITERIONS.keys()
        )
        for criterion in list_criterion:
            self.gof = CRITERIONS[criterion.lower()]
            self._criterion_nethist[criterion] = self.gof(**params)
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
            will try all of them up tp the maximum number of unique values im hist_approx.

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
        flat_graphon = hist_approx.get_theta()[np.triu_indices(nrow_tile)]
        flat_areas = hist_approx.areas[np.triu_indices(nrow_tile)]

        # Number of link communities and input check
        self._num_par_nethist = comb(nrow_tile + 1, 2)
        num_par_diff = len(np.unique(flat_graphon))
        if number_link_communities is not None and number_link_communities > num_par_diff:
            raise ValueError(
                f"The number of link communities {number_link_communities} is greater "
                + f"than the number of clusters {num_par_diff} in the histogram estimator"
            )
        clusters = (
            list(range(nrow_tile, num_par_diff + 1))
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
                labels = self._cluster(number_groups, flat_graphon)
                self.cluster_it_list.append(labels)
                avr_graphon[index] = self._smoothing_operator(
                    tensor_slice_graphon=avr_graphon[index],
                    labels=labels,
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
        self.kmeans_labels_evolution["labels"].append(kmeans.labels_)
        self.kmeans_labels_evolution["nLC"].append(number_link_communities)
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
        for j in np.unique(labels):
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
                initial_rho=hist_approx.initial_rho,
            )
            for i in range(nclust_hist)
        ]
        self.all_cluster_graphons = smoothed_graphons
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
            params = dict(
                log_likelihood_val=assignment.log_likelihood,
                n=adjacencyMatrix.shape[0],
                num_par=comb(assignment.theta.shape[0] + 1, 2),
                norm=np.linalg.norm(best_pij - adjacencyMatrix) / (n_nodes**2 - n_nodes),
                var=np.var(best_pij - adjacencyMatrix, ddof=n_nodes),
            )
            list_criterion = (
                [self.criterion_name] if self.criterion_name != "all" else CRITERIONS.keys()
            )
            for criterion in list_criterion:
                self.gof = CRITERIONS[criterion.lower()]
                self._criterion_nethist[criterion] = self.gof(**params)
                self.criterion_values[criterion] = self.gof(**params)
                self._num_par_smooth[criterion] = n_link_com
        else:
            best_graphon, best_pij = self._model_selection_from_criterion(
                tensor_graphon,
                adjacencyMatrix,
                latentVarArray,
                n_link_com,
            )
        return best_graphon, best_pij

    def _model_selection_from_criterion(
        self,
        tensor_graphon: List[StepGraphon],
        adjacencyMatrix: np.ndarray,
        latentVar: np.ndarray,
        n_link_com: List[int],
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Select the best graphon from the tensor of graphons by criterion.

        Parameters
        ----------
        tensor_graphon : List[StepGraphon]
            tensor of smoothed graphons of shape (n_graphons, n_block, n_block),
            where n_block is the number of blocks in the original fit graphon.
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        latentVar : np.ndarray
            latent variables of the graph
        n_link_com : List[int]
            number of link communities of each smoothed graphon

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            best graphon matrix and P_ij selected by the criterion, if all criterions are selected, return bic.
        """
        n_nodes = adjacencyMatrix.shape[0]
        best_graphon = tensor_graphon[0]
        best_pij = np.ones((n_nodes, n_nodes)) / 2
        list_criterion = (
            [self.criterion_name] if self.criterion_name != "all" else CRITERIONS.keys()
        )
        ref_criterion = "bic" if self.criterion_name == "all" else self.criterion_name

        for criterion in list_criterion:
            all_criterion_values = self._get_criterion_values(
                tensor_graphon, adjacencyMatrix, latentVar, n_link_com, criterion
            )
            best_index = self.choose_best(np.array(all_criterion_values), criterion)
            self._num_par_smooth[criterion] = n_link_com[best_index]
            self.criterion_values[criterion] = all_criterion_values[best_index]
            self.best_pairs[criterion]["graphon"] = tensor_graphon[best_index]
            self.best_pairs[criterion]["pij"] = tensor_graphon[best_index]._get_edge_probabilities(
                n=n_nodes, latentVarArray=latentVar
            )
        # Return by default bic to preserve estimator structure, other criterion are available in self.best_pairs.
        best_graphon = self.best_pairs[ref_criterion]["graphon"]
        best_pij = self.best_pairs[ref_criterion]["pij"]

        return best_graphon, best_pij

    def _get_criterion_values(
        self,
        tensor_graphon: List[StepGraphon],
        adjacencyMatrix: np.ndarray,
        latentVar: np.ndarray,
        n_link_com: List[int],
        criterion: str,
    ) -> List[float]:
        """Get the criterion values for each smoothed graphon.

        Parameters
        ----------
        tensor_graphon : List[StepGraphon]
            tensor of smoothed graphons of shape (n_graphons, n_block, n_block),
            where n_block is the number of blocks in the original fit graphon.
        adjacencyMatrix : np.ndarray
            adjacency matrix of the graph
        latentVar : np.ndarray
            latent variables of the graph
        n_link_com : List[int]
            number of link communities of each smoothed graphon
        criterion : str
            criterion used for the model selection, either 'aic', 'bic', 'elbow'.

        Returns
        -------
        List[float]
            list of criterion values for each smoothed graphon.
        """
        n_nodes = adjacencyMatrix.shape[0]
        all_criterion_values = []
        for i, graphon in enumerate(tensor_graphon):
            pij = graphon._get_edge_probabilities(n=n_nodes, latentVarArray=latentVar)
            log_likelihood_value = log_likelihood(pij, adjacencyMatrix)
            self.gof = CRITERIONS[criterion.lower()]
            params = dict(
                log_likelihood_val=log_likelihood_value,
                n=n_nodes,
                num_par=n_link_com[i],
                norm=np.linalg.norm(pij - adjacencyMatrix) / (n_nodes**2 - n_nodes),
                var=np.var(pij - adjacencyMatrix, ddof=n_nodes),
            )
            all_criterion_values.append(self.gof(**params))
        return all_criterion_values

    def get_criterion_value(self) -> float:
        """Get the criterion score of the estimated graphon.

        Returns
        -------
        float
            criterion score.
        """
        if self.fitted is False:
            warn("The model has not been fitted yet, returning np.inf.")
        return self.criterion_values[self.criterion_name]

    def choose_best(self, values: np.ndarray, criterion_name: Optional[str] = None, **kwargs):
        if criterion_name is None:
            criterion_name = self.criterion_name
        if criterion_name != "elbow":
            best_value = np.inf
            for i, criterion_candidate in enumerate(values):
                improvement = (
                    (best_value - criterion_candidate) / best_value if best_value != np.inf else 1
                )
                if improvement >= self._tolerance and improvement > 0:
                    best_value = criterion_candidate
                    index_best = i
        elif criterion_name == "elbow":
            index_best = elbow_point(values)
        else:
            raise ValueError(f"Criterion {criterion_name} not available.")
        return index_best

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
        for criterion in self._num_par_smooth.keys():
            self._num_par_smooth[criterion] /= self._num_par_nethist
