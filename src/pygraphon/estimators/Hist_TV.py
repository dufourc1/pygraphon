"""New implmentation of the histogram estimator smoothed with total variation."""

from typing import Optional, Tuple

import fast_tsp
import numpy as np
import scipy.stats
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
from skimage.transform import resize

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.assignment import Assignment
from pygraphon.estimators.networkhistogram.nethist import nethist
from pygraphon.graphons import StepGraphon
from pygraphon.norm import Dist, SimpleDifferenceDistance


class Hist_TV(BaseEstimator):
    """Total variation smoothed histogram estimator adapted from :cite:p:`olhede2014`."""

    def __init__(
        self,
        bandwithHist: Optional[float] = None,
        max_num_iter: int = 1000,
        tol: float = 1e-4,
        distance: Dist = SimpleDifferenceDistance(),
        tsp_precision: float = 1e-4,
        n_lambda_grid: int = 50,
        tv_algorithm: str = "bregman",
    ) -> None:
        super().__init__()
        self.bandwidthHist = bandwithHist
        self.max_num_iter = max_num_iter
        self.tol = tol
        self.distance = distance
        self.tsp_precision = tsp_precision
        self.n_lambda_grid = n_lambda_grid
        if tv_algorithm == "bregman":
            self.tv_algorithm = denoise_tv_bregman
        elif tv_algorithm == "chambolle":
            self.tv_algorithm = denoise_tv_chambolle
        else:
            raise ValueError("Invalid TV algorithm")
        self._num_par_tv = -1

    def _approximate_graphon_from_adjacency(
        self,
        adjacency_matrix: np.ndarray,
        bandwidthHist: Optional[float] = None,
        absTol: float = 2.5 * 1e-4,
        maxNumIterations: int = 500,
        past_non_improving: int = 3,
        use_default_bandwidth: bool = True,
        progress_bar: bool = False,
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
        if bandwidthHist is None:
            bandwidthHist = self.bandwidthHist
        first_fit_graphon, assignment = self._first_approximate_graphon_from_adjacency(
            adjacency_matrix,
            bandwidthHist,
            absTol,
            maxNumIterations,
            past_non_improving,
            use_default_bandwidth,
            progress_bar,
        )
        theta_matrix, col_ind = self.sort_theta(first_fit_graphon.get_theta())
        graphon_sorted = StepGraphon(theta_matrix, bandwidthHist=first_fit_graphon.bandwidthHist)

        return self.smooth(graphon_sorted, adjacency_matrix, col_ind, assignment)

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
            progress_bar=progress_bar,
        )

        bandwidthHist = h / adjacencyMatrix.shape[0]
        # ---------------------------------------------------------------------------------------------
        # Return the graphon from first fit nethist.
        graphon_nethist = StepGraphon(
            assignment_nethist.theta,
            bandwidthHist=bandwidthHist,
        )
        return graphon_nethist, assignment_nethist

    def smooth(
        self,
        graphon: StepGraphon,
        adjacency_matrix: np.ndarray,
        idx: np.ndarray,
        assignment: Assignment,
    ) -> Tuple[StepGraphon, np.ndarray]:
        """Smooth the adjacency matrix based on the permutation idx."""
        theta_matrix = graphon.get_theta()
        n = adjacency_matrix.shape[0]
        k = theta_matrix.shape[0]
        best_pred = -np.Inf
        best_sol = None
        lambda_theory = 2 * np.log(k) ** (3 / 2)

        if self.tv_algorithm == denoise_tv_bregman:  # bregman is 1/lambda
            lambda_grid = np.linspace(0.01, lambda_theory, num=self.n_lambda_grid)
        else:
            lambda_grid = 1 / np.linspace(0.01, lambda_theory, num=self.n_lambda_grid)

        for j in range(self.n_lambda_grid):
            graphon_matrix = self.tv_algorithm(
                theta_matrix,
                weight=1 / lambda_grid[j],
                max_num_iter=self.max_num_iter,
                eps=self.tol,
            )
            graphon_matrix = (graphon_matrix + graphon_matrix.T) / 2
            graphon_matrix = StepGraphon(graphon_matrix, bandwidthHist=graphon.bandwidthHist)
            pij = graphon_matrix._get_edge_probabilities(
                n=n,
                latentVarArray=assignment.labels_to_latent_variables(),
            )
            pred = np.mean((scipy.stats.binom.logpmf(adjacency_matrix, 1, pij)))
            if pred > best_pred:
                best_pred = pred
                best_sol = graphon_matrix

        # sort back
        original_indices = np.argsort(idx)
        final_theta = best_sol.get_theta()[original_indices, :]
        final_theta = final_theta[:, original_indices]
        np.clip(final_theta, 0, 1, out=final_theta)
        graphon = StepGraphon(final_theta, bandwidthHist=1 / k)
        P_hat = resize(graphon.get_theta(), (n, n), order=0, mode="edge", anti_aliasing=False)
        # remove diagonal to stay consistent with no self-loops
        P_hat[np.diag_indices_from(P_hat)] = 0

        # map to [0, 1]
        np.clip(P_hat, 0, 1, out=P_hat)

        return (
            graphon,
            P_hat,
        )

    def sort_theta(self, theta_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort the adjacency matrix based on a distance matrix."""
        col_ind = self.fast_tsp(theta_matrix)
        theta_matrix = theta_matrix[col_ind, :]
        theta_matrix = theta_matrix[:, col_ind]
        return theta_matrix, col_ind

    def fast_tsp(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Solve the traveling salesman problem using a fast heuristic."""
        distance_matrix = (distance_matrix * self.tsp_precision).astype(int)
        return fast_tsp.find_tour(distance_matrix)
