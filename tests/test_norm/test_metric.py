from pygraphon.estimators import HistogramEstimator
from pygraphon.graphons import Graphon
from pygraphon.norm import MseProbaEdge


def test_can_run():
    """Should run."""
    graphon = Graphon(function=lambda x, y: 0.5)
    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    loss = MseProbaEdge(n_nodes=10)
    loss(graphon, estimator.get_graphon())


def test_can_run_with_kwargs():
    """Should run."""
    graphon = Graphon(function=lambda x, y: 0.5)
    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    loss = MseProbaEdge(n_nodes=10)
    loss(graphon, estimator.get_graphon(), n_nodes=10)


def test_compute_Mse():
    """Should compute the MSE."""
    graphon = Graphon(function=lambda x, y: 0.5)
    loss = MseProbaEdge(n_nodes=10)
    assert loss(graphon, graphon) == 0

    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    assert loss(estimator.get_graphon(), estimator.get_graphon()) == 0
