"""Implementation of neighborhood smoothing estimator."""
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon


class NBDsmooth(BaseEstimator):
    """Estimate graphon by neighborhood smoothing :cite:p:`Zhang2015`."""

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, np.ndarray]:
        n = adjacency_matrix.shape[0]
        h = np.sqrt(np.log(n) / n)

        # compute dissimilarity matrix
        A_sq = np.dot(adjacency_matrix, adjacency_matrix) / n
        D = cdist(A_sq, A_sq, metric="minkowski", p=1)

        # compute kernel matrix
        kernel_mat = (D < np.percentile(D, q=100 * h, axis=1, keepdims=True)).astype(np.double)

        # normalize kernel matrix row-wise
        kernel_mat /= np.sum(kernel_mat, axis=1, keepdims=True) + 1e-10

        # compute P
        P_hat = np.dot(kernel_mat, adjacency_matrix)
        P_hat = (P_hat + P_hat.T) / 2

        # instantiate graphon
        graphon_hat = StepGraphon(P_hat, bandwidthHist=1 / adjacency_matrix.shape[0])

        return graphon_hat, P_hat
