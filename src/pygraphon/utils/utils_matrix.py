"""Random function to help with matrices."""
from copy import copy
from typing import Tuple

import numba as nb
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


def bound_away_from_one_and_zero_arrays(array: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Bound away from 1 and 0 in the log likelihood.

    This is done to avoid numerical issues.

    Parameters
    ----------
    array : np.ndarray
        array to bound
    eps : float
        gap to add to 0 and remove from 1, by default EPS (np.spacing(1))

    Returns
    -------
    List[np.ndarray]
        bounded arrays
    """
    array[array <= eps] = eps
    array[array >= 1 - eps] = 1 - eps
    return array


@nb.jit(nopython=True)
def scatter_symmetric_matrix(blocks: np.ndarray, group_membership: np.ndarray) -> np.ndarray:
    """Write all values from the blocks at the indices specified by group_membership.

    Will return a symmetric matrix such that the value at index (i,j) is the value of
    the block corresponding to the group_membership of i and j.

    Parameters
    ----------
    blocks : np.ndarray
        blocks of the graphon (theta matrix)
    group_membership : np.ndarray
        group membership of the nodes (n)

    Returns
    -------
    np.ndarray
        edge probability matrix (nxn)
    """
    n = group_membership.shape[0]
    P_ij = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            P_ij[i, j] = blocks[group_membership[i], group_membership[j]]
    P_ij = P_ij + P_ij.T
    return P_ij
