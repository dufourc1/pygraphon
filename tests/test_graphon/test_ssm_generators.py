import numpy as np

from pygraphon.graphons.ssm_generators import AssociativeFullyRandom, generate_hierarchical_theta


def test_theta_in_bounds():
    """Test that the theta matrix is in bounds."""
    theta = AssociativeFullyRandom(3, 0.1, 0.2, 0.3, 0.4).Theta
    assert np.all(theta <= 0.4)
    assert np.all(theta >= 0.1)


def test_theta_is_symmetric():
    """Test that the theta matrix is symmetric."""
    theta = AssociativeFullyRandom(3, 0.1, 0.2, 0.3, 0.4).Theta
    assert np.all(theta == theta.T)


def test_hierachical_theta_in_bounds():
    """Test that the hierarchical theta matrix is in bounds."""
    theta = generate_hierarchical_theta(10, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert np.all(theta <= 0.6)
    assert np.all(theta >= 0.1)


def test_hierachical_theta_is_symmetric():
    """Test that the hierarchical theta matrix is symmetric."""
    theta = generate_hierarchical_theta(10, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert np.all(theta == theta.T)
