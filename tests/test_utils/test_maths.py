# -*- coding: utf-8 -*-

"""Test of maths utils."""
from math import log

import numpy as np
import pytest

from pygraphon.utils import EPS, log_likelihood


def test_log_likelihood_simple_graph():
    """Check log-likelihood is correct."""
    probs = np.array([[0.1, 0.2, 0.3], [0.2, 0.5, 0.45], [0.3, 0.45, 0.9]])
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    theoretical_ll = log(0.2 * 0.3 * (1 - 0.45))
    assert np.isclose(log_likelihood(probs, A), theoretical_ll)


def test_log_likelihood_with_p_1_x_0():
    """Cjeck log-likelihood is correct when p=1 and x=0."""
    n = 5
    probs = np.ones((n, n)) - np.eye(n)
    a = np.zeros((n, n))
    ll = log_likelihood(probs, a)
    assert ll != -np.inf
    assert ll == log_likelihood(probs * (1 - EPS), a)


def test_log_likelihood_with_p_0_x_1():
    """Cjeck log-likelihood is correct when p=0 and x=1."""
    n = 5
    probs = np.eye(n)
    a = np.ones((n, n))
    ll = log_likelihood(probs, a)
    assert ll != -np.inf
    assert ll == log_likelihood(probs + EPS, a)


def test_diagonal_no_impact():
    """Check that the diagonal has no impact on the log-likelihood."""
    # randomly generated matrices, as we only use the
    # upper triangular part of the matrix
    n = 20
    probs = np.random.rand(n, n) - np.eye(n)
    a = np.random.randint(0, 2, (n, n))
    np.fill_diagonal(a, 0)
    a ^= a.T
    ll = log_likelihood(probs, a)
    assert ll == log_likelihood(probs + np.eye(n) * 10, a)
    assert ll == log_likelihood(probs, a + np.eye(n) * 10)


def test_log_likelihood_probs_bigger_1() -> None:
    """Check that the log-likelihood raises an error when the one of the probs are bigger than 1."""
    probs = np.array([[0, 0.2, 0.3], [0.2, 0, 1.45], [0.3, 1.45, 0]])
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        log_likelihood(probs, A)
