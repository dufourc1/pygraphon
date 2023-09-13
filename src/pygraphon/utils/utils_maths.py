"""Random maths functions."""
import math
from itertools import permutations
from typing import Iterable

import numba as nb
import numpy as np
from kneed import KneeLocator

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
    return kn.knee


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


@nb.jit(nopython=True)
def log_likelihood(probs: np.ndarray, A: np.ndarray, eps=1e-8) -> float:
    r"""Compute the log-likelihood of the graphon given the adjacency matrix of a simple graph.

    .. math::
        \sum_{i<j} \log(\theta_{ij}){A_{ij}} + \log(1-\theta_{ij}){1-A_{ij}}


    .. math::
        \theta_{ij} =  \begin{cases} \epsilon & \text{if } \theta_{ij} < \epsilon \\
        p_{ij} & \text{if } \epsilon \leq \theta_{ij} \leq 1 - \epsilon \\
        1 - \epsilon & \text{if } \theta_{ij} > 1 - \epsilon \end{cases},

    where :math:`A_{ij}` is :py:obj:`A[i,j]`, :math:`p_{ij}` is :py:obj:`probs[i,j]`,
    and :math:`\epsilon` is :py:obj:`np.spacing(1)`.

    Parameters
    ----------
    probs : np.ndarray
        edge probability matrix
    A : np.ndarray
        adjacency matrix of the graph

    Raises
    ------
    ValueError
        if the probability matrix contains values greater than 1

    Returns
    -------
    float
        log-likelihood of the graphon given the adjacency matrix

    .. note::
        Suppose :py:obj:`probs` and :py:obj:`A` are aligned, i.e. :py:obj:`probs[i,j]` is the probability
        of an edge between node :math:`i` and node :math:`j` and :py:obj:`A[i,j]` is the adjacency
        of the edge between node :math:`i` and node :math:`j`.

        This function assumes the graph is simple, i.e. only undirected edges and no self-loops.
        For this reason, only the upper triangular part of :py:obj:`probs` and :py:obj:`A` are considered.
    """
    result = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            x = A[i, j]
            p = probs[i, j]
            if p > 1:
                raise ValueError("Probability matrix contains values greater than 1")
            if p < eps:
                result += x * np.log(eps)
                result += (1 - x) * np.log(1 - eps)
            elif p > 1 - eps:
                result += x * np.log(1 - eps)
                result += (1 - x) * np.log(eps)
            else:
                result += x * np.log(p)
                result += (1 - x) * np.log(1 - p)
    return result
