"""
test_elongation.py

FINAL VERSION:
Pytest-based tests for:
- get_elongation_list(...)
- compute_average_elongation(...)
- create_symmetric_diagonal_squares_partition(...)
- create_diagonal_partition_symmetric(...)

All from utils_partitions.py

We verify:
1. Symmetry (if you enable the check in get_elongation_list).
2. Fully-labeled matrix (no zeros left).
3. Proper shape labeling {1..num_par}, no empty shapes.
4. Specific elongation properties in known scenarios.
5. Error conditions (non-consecutive labels, missing shapes, sum mismatch, etc.).
"""

import pytest
import numpy as np

from pygraphon.utils.utils_elongation import (
    get_elongation_list,
    compute_average_elongation,
    create_symmetric_diagonal_squares_partition,
    create_diagonal_partition_symmetric
)

###############################################################################
# Test for create_symmetric_diagonal_squares_partition
###############################################################################

@pytest.mark.parametrize("k", [6, 8])
def test_symmetric_diagonal_squares_partition_basic(k):
    """
    Check that create_symmetric_diagonal_squares_partition covers the entire 
    k-by-k grid with no leftover zeros, labels are consecutive, and matrix is 
    symmetric by construction.
    """
    # Example squares_sizes that sum to k: let's do 2 big squares for simplicity
    squares_sizes = [k//2, k - k//2]  # e.g. for k=6 => [3,3], for k=8 => [4,4]
    theta, num_par = create_symmetric_diagonal_squares_partition(k, squares_sizes)

    # Check shape:
    assert theta.shape == (k, k)
    # Check that the entire matrix is labeled (no zeros):
    assert not np.any(theta == 0), "No cell should remain labeled as 0."

    # Check the partition is presumably block diagonal => symmetrical
    assert np.allclose(theta, theta.T), "Matrix must be symmetric."

    # Check label count:
    distinct_labels = np.unique(theta)
    assert len(distinct_labels) == num_par, "Distinct label count must match num_par."
    # Must be labeled consecutively from 1..num_par
    expected_labels = np.arange(1, num_par+1)
    assert np.array_equal(np.sort(distinct_labels), expected_labels)

    # Now test elongation list
    elong_list = get_elongation_list(theta, num_par, k)
    assert len(elong_list) == num_par
    # No shape should be empty
    for shape_id, e_val in enumerate(elong_list, start=1):
        assert e_val > 0, f"Elongation for shape {shape_id} must be > 0."


@pytest.mark.parametrize("k, squares_sizes", [
    (6, [2,4]),
    (6, [3,3]),
    (6, [1,2,3]),
    (8, [2,2,4]),
])
def test_symmetric_diagonal_squares_partition_variations(k, squares_sizes):
    """
    Additional checks for different squares_sizes that sum to k.
    Ensures no zeros remain, no shapes are empty, etc.
    """
    theta, num_par = create_symmetric_diagonal_squares_partition(k, squares_sizes)
    # entire matrix must be labeled:
    assert np.all(theta > 0)
    # matrix must be symmetric
    assert np.allclose(theta, theta.T)

    # check labeling
    labels = np.unique(theta)
    assert len(labels) == num_par, "Mismatch in # of shapes"
    assert np.array_equal(np.sort(labels), np.arange(1, num_par+1))

    # check elongation
    elong_list = get_elongation_list(theta, num_par, k)
    assert len(elong_list) == num_par


def test_symmetric_diagonal_squares_partition_error():
    """
    If sum(squares_sizes) != k, we expect ValueError.
    """
    k = 5
    squares_sizes = [2,2]  # sum=4 != 5
    with pytest.raises(ValueError, match="does not equal k"):
        _ = create_symmetric_diagonal_squares_partition(k, squares_sizes)

###############################################################################
# Test for create_diagonal_partition_symmetric
###############################################################################

def test_diagonal_partition_symmetric_basic():
    """
    For create_diagonal_partition_symmetric(k), the matrix is fully labeled in 1..k,
    no zeros, fully symmetric, no shape is empty.
    """
    k = 5
    theta, num_par = create_diagonal_partition_symmetric(k)
    assert num_par == k, "Number of shapes must be k for diagonal partition."

    # no zeros leftover
    assert not np.any(theta == 0), "All cells must be labeled from 1..k."
    assert np.allclose(theta, theta.T), "Matrix must be symmetric."

    # check labeling
    labels = np.unique(theta)
    assert len(labels) == k, f"Should have exactly {k} distinct labels."
    expected = np.arange(1, k+1)
    assert np.array_equal(np.sort(labels), expected)

    # check elongation
    e_list = get_elongation_list(theta, num_par, k)
    assert len(e_list) == k, "One elongation per shape."

def test_diagonal_partition_symmetric_elongation():
    """
    For k=6 (for example), we expect:
      - min elongation = 2
      - max elongation = k/sqrt(k)
    if that matches your domain assumption about diagonal strips.
    """
    k = 6
    theta, num_par = create_diagonal_partition_symmetric(k)
    e_list = get_elongation_list(theta, num_par, k)

    min_elong = min(e_list)
    max_elong = max(e_list)

    # user domain knowledge
    assert pytest.approx(min_elong, 0.001) == 2.0, (
        f"Min elongation must be 2, got {min_elong:.3f}"
    )
    expected_max = k / np.sqrt(k)
    assert pytest.approx(max_elong, 0.001) == expected_max, (
        f"Max elongation must be {expected_max:.3f}, got {max_elong:.3f}"
    )

def test_diagonal_partition_symmetric_average_elongation():
    """
    If your domain knowledge says the sum or average of elongations 
    equals something like sum_{r=1..k} sqrt(r) / k, check it here.
    """
    k = 4
    theta, num_par = create_diagonal_partition_symmetric(k)
    e_list = get_elongation_list(theta, num_par, k)
    actual_sum = sum(e_list)
    expected_sum = sum(np.sqrt(r) for r in range(1, k+1))

    assert pytest.approx(actual_sum, 0.001) == expected_sum, (
        f"For k={k}, sum of elongations != sum sqrt(r)"
    )

    # average
    avg_elong = compute_average_elongation(theta, num_par, k)
    expected_avg = expected_sum / k
    assert pytest.approx(avg_elong, 0.001) == expected_avg

###############################################################################
# Test for get_elongation_list & compute_average_elongation errors
###############################################################################

def test_incorrect_num_par():
    """
    If we pass a matrix with some shapes but incorrectly claim num_par is bigger/smaller,
    should raise ValueError for distinct shapes mismatch.
    """
    k = 4
    squares_sizes = [2,2]  # sum=4 => 2 shapes
    theta, actual_num = create_symmetric_diagonal_squares_partition(k, squares_sizes)
    # but let's claim num_par=3
    with pytest.raises(ValueError, match="distinct shapes"):
        _ = get_elongation_list(theta, 3, k)

def test_non_consecutive_labels():
    """
    If we produce a matrix with shape labels {1,3} skipping 2 => error about consecutive labeling.
    """
    k = 4
    theta = np.ones((k, k), dtype=int)
    # shape label=1 everywhere except bottom-right => label=3
    theta[2:, 2:] = 3
    # We claim num_par=2. The matrix has distinct labels = {1,3} => skipping 2.
    with pytest.raises(ValueError, match="consecutively"):
        _ = get_elongation_list(theta, 2, k)

def test_empty_shape():
    """
    If shape i is not used in the matrix => empty => ValueError.
    """
    k = 4
    # We'll label everything as shape=1 => shape=2 is never used => empty
    theta = np.ones((k, k), dtype=int)
    with pytest.raises(ValueError, match="no blocks"):
        _ = get_elongation_list(theta, 2, k)

def test_symmetry_enforcement():
    """
    If you uncomment the optional check in get_elongation_list, 
    a nonsymmetric matrix should fail. We'll demonstrate how that might be tested.
    """
    k = 3
    mat = np.array([
        [1, 1, 2],
        [1, 1, 2],
        [2, 9, 1],  # '9' breaks symmetry w.r.t. mat[1,2]
    ])
    # shape labels => {1,2,9}, we claim num_par=3
    # If we enforce symmetry in get_elongation_list, this should raise an error.
    # By default, it's commented out, but if you enable it, this test should pass.
    with pytest.raises(ValueError, match="not symmetric"):
        _ = get_elongation_list(mat, 3, k)
