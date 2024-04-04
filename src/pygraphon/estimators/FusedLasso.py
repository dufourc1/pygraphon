"""Implementation of the Fused Lasso estimator."""
from typing import Optional, Tuple

import cvxpy as cp
import fast_tsp
import numpy as np
import scipy.stats

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.norm.node_distance import Dist, JensenShannonDivergenceDistance


class FusedLasso(BaseEstimator):
    """Estimate graphon by Fused Lasso (CITE PADILLA)."""

    def __init__(
        self,
        distance: Dist = JensenShannonDivergenceDistance(),
        lambda_grid: np.ndarray = 10 ** np.linspace(-1.2, 0.2, num=20),
        max_num_iter: int = 1000,
        fl_tol: float = 1e-3,
        tsp_precision: float = 1e-3,
    ):
        super().__init__()
        self.distance = distance
        self.lambda_grid = lambda_grid
        self.max_num_iter = max_num_iter
        self.fl_tol = fl_tol
        self.tsp_precision = tsp_precision

        self.llhood = None

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, lambda_grid: Optional[np.ndarray] = None
    ) -> Tuple[StepGraphon, np.ndarray]:
        n = adjacency_matrix.shape[0]
        lambda_grid = self.lambda_grid if lambda_grid is None else lambda_grid

        sorted_adj, tour = self.sort(
            distance_matrix=self.distance.calculate_distance(adjacency_matrix),
            adjacency_matrix=adjacency_matrix,
        )
        tour_inv = np.arange(n)
        tour_inv[tour] = np.arange(n)  # get the inverse permutation

        P_hat, llhood = self.two_d_bern_fl_pred(sorted_adj, lambda_grid=lambda_grid)
        self.llhood = llhood

        P_hat = P_hat[tour_inv, :][:, tour_inv]
        np.fill_diagonal(P_hat, 0)

        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])
        return graphon_hat, P_hat

    def two_d_bern_fl_pred(self, M_tour, lambda_grid):
        """Compute the best solution for the fused lasso problem.

        Parameters
        ----------
        M_tour : (n, n) ndarray
            Tour matrix.
        lambda_grid : 1d-array-like or list
            List of lambda values.

        Returns
        -------
        best_sol : (n, n) ndarray
            Array of the best solution.
        best_pred : float
            The predicted likelihood for the best solution.
        """
        n = M_tour.shape[0]
        pred = np.zeros(len(lambda_grid))

        non_missing = np.random.rand(n, n) < 0.8
        M_obs = np.where(non_missing, M_tour, 0)
        missing = ~non_missing
        indices = np.where(missing)

        best_sol = np.zeros((n, n))
        best_pred = np.Inf

        for j in range(len(lambda_grid)):
            temp = self.fused_lasso_weighted_l2_2d(M_obs, lambda_=lambda_grid[j])

            theta_hat_mat = np.array(temp).reshape((n, n))

            pred[j] = -np.sum(
                np.log(scipy.stats.binom.pmf(M_tour[indices], 1, theta_hat_mat[indices]))
            )

            if pred[j] < best_pred:
                best_pred = pred[j]
                best_sol = theta_hat_mat

        best_ind = np.argmin(pred)

        temp = self.fused_lasso_weighted_l2_2d(M_tour, lambda_=lambda_grid[best_ind])

        best_sol = temp

        return best_sol, best_pred

    def fused_lasso_weighted_l2_2d(self, y, w=None, lambda_val=1, rel_tol=None):
        """
        Solves the fused lasso on a 2D grid graph under weighted L2 loss using ADMM.

        Parameters
        ----------
        y : numpy array
            The observations.
        w : numpy array or None, optional
            The weights for the L2 loss. If None, equal weights are assumed.
        lambda_val : float, optional
            The regularization parameter for the total variation penalty.
        iter_max : int, optional
            The maximum number of iterations, by default 500.
        rel_tol : float, optional
            The relative tolerance for the solver, by default 1e-3.

        Returns
        -------
        numpy array
            The estimated graphon.
        """
        if rel_tol is None:
            rel_tol = self.fl_tol

        if w is None:
            w = np.ones(y.shape)

        # Define variables
        m, n = y.shape
        x = cp.Variable(y.shape)

        # Create the differential matrices
        Dx = cp.Variable((m, n - 1))
        Dy = cp.Variable((m - 1, n))

        # Define the objective function
        objective = cp.Minimize(
            0.5 * cp.sum(cp.multiply(w, cp.square(y - x)))
            + lambda_val * (cp.norm(Dx, 1) + cp.norm(Dy, 1))
        )

        # Constraints for differential matrices
        constraints = [Dx == x[:, 1:] - x[:, :-1], Dy == x[1:, :] - x[:-1, :]]

        # Define the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(max_iters=self.max_num_iter, rel_tol=rel_tol, verbose=True)

        return x.value

    def sort(
        self, distance_matrix: np.ndarray, adjacency_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort the adjacency matrix based on a distance matrix."""
        # get the permutation from fast TSP
        col_ind = self.fast_tsp(distance_matrix)
        A_sorted = adjacency_matrix[col_ind, :]
        A_sorted = A_sorted[:, col_ind]

        return A_sorted, col_ind

    def fast_tsp(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Solve the traveling salesman problem using a fast heuristic."""
        # scale the distance matrix to integers given the precision
        distance_matrix = (distance_matrix * self.tsp_precision).astype(int)
        return fast_tsp.find_tour(distance_matrix)
