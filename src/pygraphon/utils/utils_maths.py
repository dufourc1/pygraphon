"""Random maths functions."""
import math
from itertools import permutations
from typing import Iterable

import numpy as np
from kneed import KneeLocator
from loguru import logger

EPS = np.spacing(1)


def generate_all_permutations(size: int = 3) -> Iterable:
    """Generate all permutations of a given size.

    Parameters
    ----------
    size : int
        size of the permutation (0,1,2,...,size-1). Defaults to 3.

    Returns
    -------
    Iterable
        all permutations of the given size
    """
    return permutations(range(size))


def bic(log_likelihood_val: float, n: int, num_par: int, *args, **kwargs) -> float:
    """Compute the BIC score of the graphon.

    Parameters
    ----------
    log_likelihood_val : float
        log-likelihood of the graphon given the adjacency matrix
    n : int
        number of nodes of the graph
    num_par : int
        number of parameters of the graphon

    Returns
    -------
    float
        BIC score of the graphon
    """
    return -2 * log_likelihood_val + num_par * math.log(n * (n - 1) / 2)


def aic(log_likelihood_val: float, num_par: int, *args, **kwargs) -> float:
    """Compute the AIC score of the graphon.

    Parameters
    ----------
    log_likelihood_val : float
        log-likelihood of the graphon given the adjacency matrix
    num_par : int
        number of parameters of the graphon

    Returns
    -------
    float
        AIC score of the graphon
    """
    return -2 * log_likelihood_val + 2 * num_par


def elbow_point(norm: np.ndarray, *args, **kwargs) -> int:
    """Return the index of the elbow point of the curve.

    Parameters
    ----------
    norm : np.ndarray
        values of the norm

    Returns
    -------
    int
        index of the elbow point of the curve
    """
    x = np.arange(len(norm))
    kn = KneeLocator(x, norm, S=1, curve="convex", direction="decreasing")
    inflection_point = kn.knee
    if inflection_point is None:
        logger.warning("No elbow point found, returning minimum of the curve")
        inflection_point = np.argmin(norm)
    return inflection_point


def mallows_cp(norm: float, var: float, num_par: int, n: int, *args, **kwargs) -> float:
    """Compute the Mallows' Cp score.

    Parameters
    ----------
    norm : float
        sum of squared errors
    var : float
        variance
    num_par : int
        number of parameters
    n : int
        number of nodes of the graph

    Returns
    -------
    float
        Mallows' Cp score
    """
    return (norm + 2 * num_par * var) / n


def hqic(log_likelihood_val: float, num_par: int, n: int, *args, **kwargs) -> float:
    """Compute the HQIC score of the graphon.

    Parameters
    ----------
    log_likelihood_val : float
        log-likelihood of the graphon given the adjacency matrix
    num_par : int
        number of parameters of the graphon
    n : int
        number of nodes of the graph

    Returns
    -------
    float
        HQIC score of the graphon
    """
    return -2 * log_likelihood_val + 2 * num_par * math.log(math.log(n))


def fpe(norm: float, num_par: int, n: int, *args, **kwargs) -> float:
    """Compute the Aikake final prediction error score of the graphon.

    Parameters
    ----------
    norm : float
        sum of squared errors
    num_par : int
        number of parameters of the graphon
    n : int
        number of nodes of the graph

    Returns
    -------
    float
        FPE score of the graphon
    """
    return norm * (n + num_par + 1) / (n - num_par - 1)
