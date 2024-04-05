"""Implementation of the Fused Lasso estimator."""
from typing import Optional, Tuple

import cvxpy as cp
import fast_tsp
import numpy as np
import scipy.stats
from scipy.sparse import coo_matrix
import networkx as nx

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.norm.node_distance import Dist, JensenShannonDivergenceDistance
from pygraphon.utils import bound_away_from_one_and_zero_arrays


class FusedLasso(BaseEstimator):
    """Estimate graphon by Fused Lasso (CITE PADILLA)."""

    def __init__(
        self,
        distance: Dist = JensenShannonDivergenceDistance(),
        lambda_grid: np.ndarray = 10 ** np.linspace(-3, 2, num=20),
        max_num_iter: int = 2000,
        fl_tol: float = 1e-12,
        tsp_precision: float = 1e-4,
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
        P_hat = (P_hat + P_hat.T) / 2
        np.clip(P_hat, a_min=0, a_max=1, out=P_hat)

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
        # take 80% for training
        non_missing = np.random.rand(n, n) < 0.8
        M_obs = np.where(non_missing, M_tour, 0)
        indices_to_eval = np.where(~non_missing)

        best_sol = np.zeros((n, n))
        best_pred = np.Inf
        best_ind = None

        for j in range(len(lambda_grid)):
            temp = self.fused_lasso_weighted_l2_2d(M_obs, lambda_val=lambda_grid[j])

            theta_hat_mat = bound_away_from_one_and_zero_arrays(np.array(temp).reshape((n, n)))

            pred[j] = -np.mean(
                np.log(scipy.stats.binom.pmf(M_tour[indices_to_eval], 1, theta_hat_mat[indices_to_eval]))
            )

            if pred[j] < best_pred:
                best_pred = pred[j]
                best_ind = j

        print("Best lambda: ", lambda_grid[best_ind], "best pred: ", best_pred)

        best_sol = self.fused_lasso_weighted_l2_2d(M_tour, lambda_val=lambda_grid[best_ind])
        best_pred = -np.sum(
                np.log(scipy.stats.binom.pmf(M_tour, 1, best_sol))
            )

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

        y.dtype = np.float64

        # Define variables
        m, n = y.shape
        D = construct_incidence_matrix_networkx(n).T

        # Optimization problem
        beta = cp.Variable((n, n), symmetric=True)
        objective = cp.Minimize((0.5 * cp.norm(y - beta, 'fro')**2 + lambda_val * cp.norm(D @ cp.vec(beta), 1))/n**2)
        problem = cp.Problem(objective)

        # Solve the problem
        problem.solve()        
        return beta.value

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


def construct_incidence_matrix_networkx(n: int) -> np.ndarray:
    """Construct the incidence matrix for a 2D grid graph.

    Parameters
    ----------
    n : int
        The number of nodes in each dimension of the grid.

    Returns
    -------
    numpy.ndarray
        The incidence matrix.
    """
    D_col = np.zeros((n, n))
    np.fill_diagonal(D_col[:,1:], 1)
    np.fill_diagonal(D_col, -1)
    D_col = D_col[:, :-1]  # Adjust shape to perform the intended operation

    return D_col