import pytest

from pygraphon.estimators import HistogramEstimator
from pygraphon.graphons import Graphon
from pygraphon.norm import MseProbaEdge


def test_cannot_use_loss_estimator_not_fitted():
    """Should raise an error if the estimator is not fitted."""
    graphon = Graphon(function=lambda x, y: 0.5)
    estimator = HistogramEstimator()
    loss = MseProbaEdge(n_nodes=10)
    with pytest.raises(ValueError):
        loss(graphon, estimator)


def test_can_run():
    """Should run."""
    graphon = Graphon(function=lambda x, y: 0.5)
    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    loss = MseProbaEdge(n_nodes=10)
    loss(graphon, estimator)
