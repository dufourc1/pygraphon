"""Random maths functions."""
import math
from itertools import permutations
from typing import Iterable

import numpy as np
from scipy.stats import bernoulli

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


def bic(log_likelihood_val: float, n: int, num_par: int) -> float:
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


def log_likelihood(probs: np.ndarray, A: np.ndarray) -> float:
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
    return bernoulli.logpmf(np.triu(A, k=1), np.clip(np.triu(probs, k=1), EPS, 1 - EPS)).sum()
