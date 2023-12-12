# -*- coding: utf-8 -*-

"""Test of graphons classes."""

import math

import numpy as np
import pytest

from pygraphon.graphons import common_graphons as cgf
from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils import edge_density


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
        P_ij = graphon.get_edge_probabilities(n=20, wholeMatrix=False)
        assert np.all(P_ij <= 1)
        assert np.all(P_ij >= 0)

    def test_edge_probability_correct(self, graphon, function):
        P_small = graphon.get_edge_probabilities(n=4, wholeMatrix=True, exchangeable=False)
        P_theoric = np.array([function(x / 4, y / 4) for x in range(4) for y in range(4)]).reshape(
            4, 4
        )
        np.fill_diagonal(P_theoric, 0)
        assert np.allclose(P_small, P_theoric)


def test_no_initial_rho_given():
    """Test that the initial rho is computed."""
    graphon = Graphon(lambda x, y: 0.5)
    assert math.isclose(graphon.initial_rho, 0.5)
    with pytest.raises(ValueError):
        graphon = Graphon(lambda x, y: 0.5, initial_rho=-1)


def test_incompatible_initial_rho():
    """Test that a warning is raised if the initial rho is not compatible."""
    graphon = Graphon(lambda x, y: 0.5, initial_rho=0.2)
    assert math.isclose(graphon.initial_rho, 0.5)
    P = graphon.get_edge_probabilities(n=20, wholeMatrix=False)
    assert np.all(P == 0.5)


def test_initial_rho_with_normalized_f():
    """Test that initial rho is not modified if close but different."""
    initial_rho = 0.3
    graphon = Graphon(lambda x, y: 1, initial_rho=initial_rho)
    assert initial_rho == graphon.initial_rho
    P = graphon.get_edge_probabilities(n=20, wholeMatrix=False)
    assert np.all(P == initial_rho)


def test_draw_rho():
    """Test that the draw rho is correct."""
    graphon = Graphon(lambda x, y: 1)
    n = 1000
    np.random.seed(0)
    A = graphon.draw(n=n, rho=0.8)
    assert math.isclose(edge_density(A), 0.8, abs_tol=0.01)
    A = graphon.draw(n=n, rho=0.2)
    assert math.isclose(edge_density(A), 0.2, abs_tol=0.01)
    A = graphon.draw(n=n)
    assert edge_density(A) == 1


def test_integral():
    """Test that the integral methods are correct."""
    for _, graphon in cgf.items():
        first_integral = graphon.integral()
        graphon.integration_method = "simpson"
        second_integral = graphon.integral()
        assert math.isclose(first_integral, second_integral, abs_tol=1e-3)
        assert math.isclose(first_integral, 1, abs_tol=1e-3)
        assert math.isclose(second_integral, 1, abs_tol=1e-3)

        assert math.isclose(
            graphon.unormalized_graphon_function(0.2, 0.3),
            graphon.graphon_function(0.2, 0.3) * graphon.initial_rho,
        )


def test_error_on_not_implemented_integration():
    """Test that an error is raised if the integration method is not implemented."""
    with pytest.raises(ValueError):
        Graphon(lambda x, y: 1, integration_method="not_implemented")

    graphon = Graphon(lambda x, y: 1)
    graphon.integration_method = "not_implemented"
    with pytest.raises(ValueError):
        graphon.integral()


def test_print_method(capfd):
    """Test that the print method is correct."""
    g = Graphon(lambda x, y: 1)
    assert str(g) == "g = Graphon(lambda x, y: 1)"
    print(g)  # noqa: T201
    out, _ = capfd.readouterr()
    assert out == "g = Graphon(lambda x, y: 1)\n"
    g2 = Graphon(lambda x, y: 1, repr="Erdos-Renyi")
    assert str(g2) == "Erdos-Renyi"
    print(g2)  # noqa: T201
    out, _ = capfd.readouterr()
    assert out == "Erdos-Renyi\n"


def test_sparsification():
    """Test that the sparsification method is correct."""
    g = Graphon(lambda x, y: 1)
    assert g.initial_rho == 1
    g.sparsify(0.5)
    assert g.initial_rho == 0.5
    with pytest.raises(ValueError):
        g.sparsify(2)
