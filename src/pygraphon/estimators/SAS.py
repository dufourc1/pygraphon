"""Implementation of the matrix completion scheme estimator."""
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_bregman
from skimage.transform import resize

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon


class SAS(BaseEstimator):
    """Estimate graphons by sorting and smoothing :cite:p:`chan2014`."""

    def __init__(
        self,
        h: Optional[Union[int, float]] = None,
        mu: float = 7.5,
        max_num_iter: int = 20,
        tol: float = 1e-3,
        isotropic: bool = True,
    ):
        super().__init__()
        self.h = h
        self.mu = mu
        self.max_num_iter = max_num_iter
        self.tol = tol
        self.isotropic = isotropic

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray, h=None
    ) -> Tuple[StepGraphon, np.ndarray]:
        n = adjacency_matrix.shape[0]
        h = self._decide_h(n=n, h=h)

        A, idx = self.empirical_degree_sorting(adjacency_matrix)

        # histogram estimation
        H = convolve2d(A, np.ones((h, h)) / (h**2), mode="same", boundary="symm")[::h, ::h]

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

        # map to [0, 1]
        np.clip(P_hat, 0, 1, out=P_hat)
        np.clip(graphon_matrix, 0, 1, out=graphon_matrix)

        return (
            StepGraphon(graphon_matrix, bandwidthHist=1 / graphon_matrix.shape[0]),
            P_hat,
        )

    def empirical_degree_sorting(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort the adjacency matrix by the empirical degree.

        Sort in descending order.

        Args:
            A (np.ndarray): Adjacency matrix

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            np.ndarray: Sorted adjacency matrix
            np.ndarray: Index of the sorted adjacency matrix
        """
        # sort in descending order
        idx = np.argsort(np.sum(A, axis=0))[::-1]
        A_sorted = A[idx, :]
        A_sorted = A_sorted[:, idx]
        return A_sorted, idx

    def _decide_h(self, n: int, h: Optional[Union[int, float]] = None) -> int:
        """Decide the value of h based on algorithm paramemters and inpput.

        Returns
        -------
        int
            Value of h
        """
        if h is None:
            if self.h is None:
                h = int(np.log(n))
            else:
                if type(self.h) == float:
                    h = int(self.h * n)
                else:
                    h = self.h
        return h  # type: ignore
