import numpy as np
import sklearn

from pygraphon.estimators import HistogramEstimator
from pygraphon.graphons import Graphon
from pygraphon.norm import AUCEdge, AUPRCEdge, MaeProbaEdge, MseProbaEdge


class GraphonDeterministic(Graphon):
    def draw(self, n: int) -> np.ndarray:
        edge_probs = self.get_edge_probabilities(n, False)
        return np.array(edge_probs > 0.5, dtype=int)


def test_compute_Mse_with_self():
    """Should compute the MSE."""
    graphon = Graphon(function=lambda x, y: 0.5)
    loss = MseProbaEdge(n_nodes=10)
    assert loss(graphon, graphon) == 0

    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    assert loss(estimator.get_graphon(), estimator.get_graphon()) == 0


def test_compute_Mae_with_self():
    """Should compute the MAE."""
    graphon = Graphon(function=lambda x, y: 0.5)
    loss = MaeProbaEdge(n_nodes=10)
    assert loss(graphon, graphon) == 0

    estimator = HistogramEstimator()
    A = graphon.draw(rho=1, n=30)
    estimator.fit(A)
    assert loss(estimator.get_graphon(), estimator.get_graphon()) == 0


def test_compute_AUC():
    """Should compute the AUC."""
    metric = AUCEdge()
    graphon = GraphonDeterministic(lambda x, y: x * y)
    A = graphon.draw(n=300)
    # As the diagonal is 0 and pij is symmetric, we can flatten the matrix.
    theoretical_AUC = sklearn.metrics.roc_auc_score(
        A.flatten(), graphon.get_edge_probabilities(300, False, True).flatten()
    )

    assert metric(A, graphon.get_edge_probabilities(300, False, True)) == theoretical_AUC


def test_compute_AUPRC():
    """Should compute the AUPRC."""
    metric = AUPRCEdge()
    graphon = GraphonDeterministic(lambda x, y: x * y)
    A = graphon.draw(n=10)
    # As the diagonal is 0 and pij is symmetric, we can flatten the matrix.
    theoretical_AUPRC = sklearn.metrics.average_precision_score(
        A.flatten(), graphon.get_edge_probabilities(10, False, True).flatten()
    )

    assert metric(A, graphon.get_edge_probabilities(10, False, True)) == theoretical_AUPRC
