# -*- coding: utf-8 -*-

"""Test of matrices utils function."""

import numpy as np

from pygraphon.utils.utils_matrix import (
    bound_away_from_one_and_zero_arrays,
    check_symmetric,
    permute_matrix,
    upper_triangle_values,
)


def test_permute_matrix_simple():
    """Check permutation is correct for simple 2x2 matrix."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[4, 3], [2, 1]])
    assert np.allclose(permute_matrix(a, (1, 0)), b)


def test_permute_matrix_complex():
    """Check permutation is correct for 3x3 matrix."""
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    assert np.allclose(permute_matrix(a, (2, 1, 0)), b)


def test_check_symmetric():
    """Check that the symmetric test is correct."""
    a = np.array([[1, 2], [2, 1]])
    assert check_symmetric(a)


def test_upper_triangle_values():
    """Check that the upper triangle values are correct."""
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.allclose(upper_triangle_values(a), [1, 2, 3, 5, 6, 9])


def test_bound_away():
    """Check that the bound away from one and zero arrays is correct."""
    a = np.array([[0, 0.2, 0.3], [0.4, 0.5, 0.6], [1, 0.8, 4]])
    bounded = bound_away_from_one_and_zero_arrays(a)
    assert np.all(bounded > 0)
    assert np.all(bounded < 1)
