# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import math

import numpy as np
import pytest

from pygraphon.graphons.Graphon import Graphon


@pytest.fixture(scope="class")
def function():
    """Create a theta matrix."""
    return lambda x, y: (x + y) / 2


@pytest.fixture(scope="class")
def graphon(function):
    """Create a step graphon."""
    return Graphon(function=function)


class TestGraphon:
    """Test a graphon."""

    def test_instantiation(self, graphon, function):
        """Test that the step graphon is correctly instantiated."""
        # compute theoretical quantities
        integral = 0.5
        assert math.isclose(graphon.initial_rho, integral)

    def test_normalized(self, graphon):
        """Test that the step graphon is normalized."""
        assert math.isclose(graphon.integral(), 1)

    def test_edge_probability_between_0_and_1(self, graphon):
        """Test that the edge probability is correct."""
        P_ij = graphon._get_edge_probabilities(n=20, wholeMatrix=False)
        assert np.all(P_ij <= 1)
        assert np.all(P_ij >= 0)

    def test_edge_probability_correct(self, graphon, function):
        P_small = graphon._get_edge_probabilities(n=4, wholeMatrix=True, exchangeable=False)
        P_theoric = np.array([function(x / 4, y / 4) for x in range(4) for y in range(4)]).reshape(
            4, 4
        )
        np.fill_diagonal(P_theoric, 0)
        assert np.allclose(P_small, P_theoric)


def test_no_initial_rho_given():
    """Test that the initial rho is computed."""
    graphon = Graphon(lambda x, y: 0.5)
    assert math.isclose(graphon.initial_rho, 0.5)


def test_raise_warning_incompatible_initial_rho():
    """Test that a warning is raised if the initial rho is not compatible."""
    with pytest.warns(UserWarning):
        graphon = Graphon(lambda x, y: 0.5, initial_rho=0.2)
    assert math.isclose(graphon.initial_rho, 0.5)
    P = graphon._get_edge_probabilities(n=20, wholeMatrix=False)
    assert np.all(P == 0.5)