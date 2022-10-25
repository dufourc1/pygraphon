"""Random function to help with matrices."""
from copy import copy
from typing import List, Tuple

import numpy as np

EPS = np.spacing(1)


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
    """Check if a matrix is symmetric.

    Parameters
    ----------
    a : np.ndarray
        array to check
    rtol : float
        relative tolerance. Defaults to 1e-05.
    atol : float
        absolute tolerance. Defaults to 1e-08.

    Returns
    -------
    bool
        True if the array is symmetric, False otherwise.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def permute_matrix(matrix: np.ndarray, permutation: Tuple) -> np.ndarray:
    """Permute a matrix according to a permutation.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to permute
    permutation : Tuple
         permutation to apply: (3,1,2,0)  is interpreted as (0,1,2,3) -> (3,1,2,0) (only swap the first and last
         indices)

    Returns
    -------
    np.ndarray
        permuted matrix
    """
    new_matrix = copy(matrix)
    new_matrix[permutation, :] = matrix[:, permutation]
    return new_matrix


def upper_triangle_values(array):
    """Return the upper triangle values of an array (including the diagonal).

    Parameters
    ----------
    array : np.ndarray
         original array

    Returns
    -------
    np.ndarray
         upper triangle values
    """
    return array[np.triu_indices(array.shape[0])]


def bound_away_from_one_and_zero_arrays(
    arrays: List[np.ndarray], eps: float = EPS
) -> List[np.ndarray]:
    """Bound away from 1 and 0 in the log likelihood.

    This is done to avoid numerical issues.

    Parameters
    ----------
    arrays : List[np.ndarray]
        list of arrays to bound
    eps : float
        gap to add to 0 and remove from 1, by default EPS (np.spacing(1))

    Returns
    -------
    List[np.ndarray]
        bounded arrays
    """
    for array in arrays:
        array[array <= 0] = eps
        array[array >= 1] = 1 - eps
    return arrays
