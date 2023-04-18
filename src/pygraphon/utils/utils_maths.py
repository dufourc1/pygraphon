"""Random maths functions."""
import math
from itertools import permutations
from typing import Iterable

import numpy as np

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


def log_likelihood(probs: np.ndarray, A: np.ndarray, eps: float = EPS) -> float:
    """Compute the log-likelihood of the graphon given the adjacency matrix.

    Suppose probs and A are aligned.

    Parameters
    ----------
    probs : np.ndarray
        edge probability matrix
    A : np.ndarray
        adjacency matrix of the graph
    eps : float
        small value to avoid numerical issues with log(0)

    Returns
    -------
    float
        log-likelihood of the graphon given the adjacency matrix
    """
    probs_stable = np.clip(probs, eps, 1 - eps)
    return np.sum(A * np.log(probs_stable) + (1 - A) * np.log(1 - probs_stable))
