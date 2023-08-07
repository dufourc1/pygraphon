"""Test matrix completion estimator."""

import numpy as np
import pytest

from pygraphon.estimators import Completion
from pygraphon.graphons import common_graphons


@pytest.mark.parametrize("graphon", common_graphons.values())
@pytest.mark.parametrize("n", [33, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_graphon(graphon, n, seed):
    """Test that the matrix cimpletion estimator does not throw an error."""
    np.random.seed(seed)
    estimator = Completion()
    g = graphon.draw(n)
    estimator.fit(g)
