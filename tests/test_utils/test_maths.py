# -*- coding: utf-8 -*-

"""Test of maths utils."""
import numpy as np

from pygraphon.utils.utils_maths import log_likelihood


def test_log_likelihood():
    """Check log-likelihood is correct."""
    probs = np.array([[0.1, 0.2, 0.3], [0.2, 0.5, 0.45], [0.3, 0.45, 0.9]])
    A = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]])
    assert np.isclose(
        log_likelihood(probs, A),
        np.log(0.9)+np.log(0.2)+np.log(0.3)+np.log(0.2)+np.log(0.5)+np.log(0.55)+np.log(0.3)+np.log(0.55)+np.log(0.9)
    )
