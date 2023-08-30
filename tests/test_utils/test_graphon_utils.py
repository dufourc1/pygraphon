# -*- coding: utf-8 -*-

"""Test of graphon utils functions."""

import numpy as np

from pygraphon.utils.utils_graphon import (
    check_consistency_graphon_shape_with_bandwidth,
    compute_areas_histogram,
)


def test_compute_areas_histogram():
    """Check that the areas of the histogram are correct."""
    bandwidth = 0.4
    theoretical_areas = np.array(
        [
            [bandwidth**2, bandwidth**2, 0.2 * 0.4],
            [bandwidth**2, bandwidth**2, 0.2 * 0.4],
            [0.2 * 0.4, 0.2 * 0.4, 0.2**2],
        ]
    )
    theta = np.ones((3, 3))
    areas = compute_areas_histogram(theta, bandwidth)
    assert np.allclose(areas, theoretical_areas)


def test_compute_areas_histogram_notzero():
    """Check that the areas of the histogram are correct when the remainder is zero."""
    bandwidth = 1 / 3
    theoretical_areas = bandwidth**2 * np.ones((3, 3))
    theta = np.ones((3, 3))
    areas = compute_areas_histogram(theta, bandwidth)
    assert np.allclose(areas, theoretical_areas)


def test_check_consistency_bandwidth_shape_graphon():
    """Test that the bandwidth and the shape of the graphon matrix are consistent."""
    for n in range(3, 2000, 14):
        bandwidth = 1 / n
        check_consistency_graphon_shape_with_bandwidth((n, n), bandwidth)
