"""Random functions for graphon."""

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
