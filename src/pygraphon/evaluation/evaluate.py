"""Base class for evaluating the performance of an estimator."""
from abc import abstractclassmethod
from typing import List

import numpy as np
from tqdm import tqdm

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon
from pygraphon.norm import BaseMetric


class Evaluate:
    """Base class for evaluating the performance of an estimator."""

    def __init__(self, graphon: Graphon, metrics: List[BaseMetric], progress_bar: bool = True):
        self.graphon = graphon
        self.metrics = metrics
        self.progress_bar = progress_bar

    @abstractclassmethod
    def evaluate(
        self, estimator: BaseEstimator, n_nodes: int, n_repetitions: int
    ) -> List[np.ndarray]:
        """Evaluate the performance of the estimator.

        Will use :py:obj:`self.graphon` to generate :py:obj:`n_repetitions` graphs
        with :py:obj:`n_nodes` nodes and then report the metric values for each
        graph and metrics.

        Parameters:
        ----------
            n_samples: int
                number of nodes in the graph
            n_repetitions: int
                number of times to run the evaluation

        Returns
        -------
        List[np.ndarray]
            List of length :py:obj:`len(self.metrics)` containing the results
            for each metric. Each element of the list is a numpy array of
            length :py:obj:`n_repetitions` containing the metric values for
            each graph.
        """
        results = [np.zeros(n_repetitions) for _ in range(len(self.metrics))]
        for i in tqdm(range(n_repetitions), disable=not self.progress_bar):
            graph = self.graphon.draw(rho=1.0, n=n_nodes, exchangeable=True)
            estimator.fit(graph)
            for j, metric in enumerate(self.metrics):
                results[j][i] = metric(graphon=self.graphon, estimator=estimator, adj_matrix=graph)

        return results

    @abstractclassmethod
    def _save_resutls(self, results, path):
        pass
