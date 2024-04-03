"""Implementation of the graphon Lasso defusing estimator from (CITE)."""

from typing import Optional, Tuple, Union

import fast_tsp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_bregman
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
        max_num_iter: int = 20,
        tol: float = 1e-3,
        isotropic: bool = True,
        distance: Dist = SimpleDifferenceDistance(),
        tsp_precision: float = 1e-3,
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
        h = self._decide_h(n=n, h=h)
        # histogram estimation
        H = convolve2d(adjacency_matrix, np.ones((h, h)) / (h**2), mode="same", boundary="symm")[
            ::h, ::h
        ]

        # total variation minimization
        n_half = round(n / 2)
        Hpad = np.pad(H, ((n_half, n_half), (n_half, n_half)), mode="symmetric")

        graphon_matrix = denoise_tv_bregman(
            Hpad,
            weight=2 * self.mu,
            max_num_iter=self.max_num_iter,
            eps=self.tol,
            isotropic=self.isotropic,
        )

        # crop to remove padding
        graphon_matrix = graphon_matrix[n_half:-n_half, n_half:-n_half]

        # resize the graphon to size of the adjacency matrix
        P_hat = resize(graphon_matrix, (n, n), order=0, mode="edge", anti_aliasing=False)

        # sort back
        original_indices = np.argsort(idx)
        P_hat = P_hat[original_indices, :]
        P_hat = P_hat[:, original_indices]
        # remove diagonal to stay consistent with no self-loops
        P_hat[np.diag_indices_from(P_hat)] = 0

        # map to [0, 1]
        np.clip(P_hat, 0, 1, out=P_hat)
        np.clip(graphon_matrix, 0, 1, out=graphon_matrix)

        return P_hat, graphon_matrix

    def fast_tsp(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Solve the traveling salesman problem using a fast heuristic."""
        # scale the distance matrix to integers given the precision
        distance_matrix = (distance_matrix * self.tsp_precision).astype(int)
        return fast_tsp.find_tour(distance_matrix)
