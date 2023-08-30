"""Random functions for graphon."""

import math
from typing import Tuple

import numpy as np


def compute_areas_histogram(theta, bandwidth):
    """Compute the areas of the histogram.

    Parameters
    ----------
    theta : np.ndarray
        theta matrix
    bandwidth : float
        bandwidth of a regular histogram, size of the blocks (0,1]x(0,1] (e.g. 0.1)

    Returns
    -------
    areas : np.ndarray
        areas of the histogram's blocks
    """
    areas = np.ones_like(theta) * bandwidth**2
    remainder = 1 - int(1 / bandwidth) * bandwidth
    if remainder != 0:
        areas[:, -1] = bandwidth * remainder
        areas[-1, :] = bandwidth * remainder
        areas[-1, -1] = remainder**2
    return areas


def check_consistency_graphon_shape_with_bandwidth(
    graphon_shape: Tuple[int, int], bandwidth: float, decimal_tol: int = 10
):
    """Check that the graphon matrix has the correct shape given the bandwidth.

    Parameters
    ----------
    graphon_shape : Tuple[int, int]
        shape of the graphon matrix
    bandwidth : float
        bandwidth of a regular histogram, size of the blocks (0,1]x(0,1] (e.g. 0.1)
    decimal_tol : int
        number of decimals to round the inverse of the bandwidth for stability

    Raises
    ------
    ValueError
        if the graphon matrix has not the correct shape given the bandwidth
    """
    inverse_bandwidth = 1 / bandwidth
    expected_shape = int(math.ceil(np.round(inverse_bandwidth, decimals=decimal_tol)))
    if graphon_shape[0] != expected_shape or graphon_shape[1] != expected_shape:
        error_msg = "The graphon matrix should have size consistent with the bandwidth. "
        details = f"graphon.shape[0] = {graphon_shape[0]}, bandwidthHist = {bandwidth}"
        choice_best = f" expexted shape {expected_shape} from {inverse_bandwidth}"
        raise ValueError(error_msg + details + choice_best)
