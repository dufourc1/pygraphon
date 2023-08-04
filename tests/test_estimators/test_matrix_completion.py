"""Test matrix completion estimator."""

import numpy as np
from pygraphon.estimators import Completion
from pygraphon.graphons import graphon_product


def test_product_graphon():
    estimator = Completion()
    n = 1000
    g = graphon_product.draw(n)
    estimator.fit(g)
