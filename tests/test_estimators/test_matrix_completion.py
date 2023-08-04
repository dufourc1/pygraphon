"""Test matrix completion estimator."""

import pytest
import numpy as np
from pygraphon.estimators import Completion
from pygraphon.graphons import common_graphons


@pytest.mark.parametrize("graphon", common_graphons.values())
@pytest.mark.parametrize("n", [33, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_product_graphon(graphon, n, seed):
    np.random.seed(seed)
    estimator = Completion()
    g = graphon.draw(n)
    estimator.fit(g)
