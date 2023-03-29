from typing import Tuple

import numpy as np
import pytest

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.evaluation.evaluate import Evaluate
from pygraphon.graphons import Graphon
from pygraphon.norm import BaseMetric


class DummyEstimator(BaseEstimator):
    def _approximate_graphon_from_adjacency(self, adjacency_matrix) -> Tuple[Graphon, np.ndarray]:
        return Graphon(lambda x, y: 0.5), np.ones_like(adjacency_matrix) * 0.5 - np.eye(
            adjacency_matrix.shape[0]
        )


class DummyMetric(BaseMetric):
    def _compute(self, graphon, estimated, adjacency_matrix):
        return 0.5

    def __str__(self):
        return "Dummy metric"


@pytest.fixture
def graphon():
    return Graphon(lambda x, y: 0.5)


@pytest.mark.parametrize(
    "metrics", [[DummyMetric()], [DummyMetric(), DummyMetric()], DummyMetric()]
)
@pytest.mark.parametrize("n_nodes", [15, [15, 14]])
def test_evaluate(graphon, metrics, n_nodes):
    evaluator = Evaluate(graphon=graphon, metrics=metrics, progress_bar=False)
    results = evaluator.evaluate(estimator=DummyEstimator(), n_nodes=n_nodes, n_repetitions=7)
    lenght_nodes = 1 if isinstance(n_nodes, int) else len(n_nodes)
    assert results[0].shape == (lenght_nodes, 7)
    assert len(results) == len(evaluator.metrics)
    assert np.mean(results[0]) == 0.5
