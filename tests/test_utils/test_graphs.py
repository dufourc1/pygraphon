# -*- coding: utf-8 -*-

"""Test of matrices utils graphs."""

import networkx as nx
import numpy as np
import pytest

from pygraphon.utils.utils_graph import (
    check_simple_adjacency_matrix,
    edge_density,
    get_adjacency_matrix_from_graph,
)


def test_get_adjacency_matrix_from_graph():
    """Check that the adjacency matrix is correct."""
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    graph = nx.from_numpy_array(a)
    assert np.allclose(get_adjacency_matrix_from_graph(graph), a)


def test_edge_densitx():
    """Check that the edge density is correct."""
    g = nx.erdos_renyi_graph(100, 0.4)
    edge_density_true = nx.density(g)
    edge_density_computed = edge_density(get_adjacency_matrix_from_graph(g))
    assert edge_density_true == edge_density_computed


def test_check_simple_adj_matrix():
    """Check that the check_simple_adjacency_matrix function works by testtng bad cases."""
    # matrix not empty on diagonal
    a = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)

    # matrix not binary
    a = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)

    # matrix not symmetric
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)

    # matrix not square
    a = np.array([[0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)

    # matrix not 2D
    a = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)
    b = np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(b)

    # matrix at least 2x2
    a = np.array([[0]])
    with pytest.raises(ValueError):
        check_simple_adjacency_matrix(a)

    # check that the function works with a correct matrix
    a = nx.adjacency_matrix(nx.complete_graph(2)).todense()
    check_simple_adjacency_matrix(a)


if __name__ == "__main__":
    a = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    check_simple_adjacency_matrix(a)
