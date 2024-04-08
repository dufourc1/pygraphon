"""Implementation of the graphon Lasso defusing estimator from (CITE)."""

from typing import Optional, Tuple, Union

import fast_tsp
import numpy as np
import scipy.stats
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
from skimage.transform import resize

from pygraphon.estimators.SAS import SAS
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.norm import Dist, SimpleDifferenceDistance


class TV_FL(SAS):
    """Estimate graphons by solving a total variation fused lasso problem.
    This is based on the SAS estimator where we, instead, define another sorting of the adjacency matrix
    based on a distance matrix and then obtain a permutation by solving a traveling salesman problem.
    """

    def __init__(
        self,
        h: Optional[Union[int, float]] = None,
        mu: float = 7.5,
        max_num_iter: int = 1000,
        tol: float = 1e-4,
        isotropic: bool = True,
        distance: Dist = SimpleDifferenceDistance(),
        tsp_precision: float = 1e-4,
        n_lambda_grid: int = 50,
        tv_algorithm: str = "bregman",
    ):
        """
        Parameters
        ----------
        distance : Dist, optional
            The distance metric to use for sorting the adjacency matrix, by default SimpleDifferenceDistance()
        tsp_precision : float, optional
            The precision to use for the traveling salesman problem, by default 1e-3
        """
        super().__init__(h, mu, max_num_iter, tol, isotropic)
        self.distance = distance
        self.tsp_precision = tsp_precision
        self.n_lambda_grid = n_lambda_grid
        if tv_algorithm == "bregman":
            self.tv_algorithm = denoise_tv_bregman
        elif tv_algorithm == "chambolle":
            self.tv_algorithm = denoise_tv_chambolle
        else:
            raise ValueError("Invalid TV algorithm")

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, h=None
    ) -> Tuple[StepGraphon, np.ndarray]:
        A, idx = self.sort(
            distance_matrix=self.distance.calculate_distance(adjacency_matrix),
            adjacency_matrix=adjacency_matrix,
        )

        P_hat, graphon_matrix = self.smooth(A, idx)

        return (
            StepGraphon(graphon_matrix, bandwidthHist=1 / graphon_matrix.shape[0]),
            P_hat,
        )

    def sort(
        self, distance_matrix: np.ndarray, adjacency_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort the adjacency matrix based on a distance matrix."""
        # get the permutation from fast TSP
        col_ind = self.fast_tsp(distance_matrix)
        A_sorted = adjacency_matrix[col_ind, :]
        A_sorted = A_sorted[:, col_ind]

        return A_sorted, col_ind

    def smooth(self, adjacency_matrix: np.ndarray, idx: np.ndarray, h=None) -> np.ndarray:
        """Smooth the adjacency matrix based on the permutation idx."""
        n = adjacency_matrix.shape[0]
        best_pred = -np.Inf
        best_sol = np.zeros((n, n))
        lambda_theory = 2 * np.log(n) ** (3 / 2)
        if self.tv_algorithm == denoise_tv_bregman:  # bregman is 1/lambda
            lambda_grid = np.linspace(0.01, lambda_theory, num=self.n_lambda_grid)
        else:
            lambda_grid = 1 / np.linspace(0.01, lambda_theory, num=self.n_lambda_grid)
        for j in range(self.n_lambda_grid):
            graphon_matrix = self.tv_algorithm(
                adjacency_matrix,
                weight=1 / lambda_grid[j],
                max_num_iter=self.max_num_iter,
                eps=self.tol,
            )
            # get the likelihood
            pred = np.mean((scipy.stats.binom.logpmf(adjacency_matrix, 1, graphon_matrix)))
            if pred > best_pred:
                best_pred = pred
                best_sol = graphon_matrix

        # resize the graphon to size of the adjacency matrix
        P_hat = resize(best_sol, (n, n), order=0, mode="edge", anti_aliasing=False)

        # sort back
        original_indices = np.argsort(idx)
        P_hat = P_hat[original_indices, :]
        P_hat = P_hat[:, original_indices]
        # remove diagonal to stay consistent with no self-loops
        P_hat[np.diag_indices_from(P_hat)] = 0

        # map to [0, 1]
        np.clip(P_hat, 0, 1, out=P_hat)
        np.clip(best_sol, 0, 1, out=best_sol)

        return P_hat, best_sol

    def fast_tsp(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Solve the traveling salesman problem using a fast heuristic."""
        # scale the distance matrix to integers given the precision
        distance_matrix = (distance_matrix * self.tsp_precision).astype(int)
        return fast_tsp.find_tour(distance_matrix)
