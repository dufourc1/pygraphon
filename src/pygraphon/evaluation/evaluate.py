"""Base class for evaluating the performance of an estimator."""
from typing import List, Union

import numpy as np
from tqdm import tqdm

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon
from pygraphon.norm import BaseMetric


class Evaluate:
    """Base class for evaluating the performance of an estimator."""

    def __init__(
        self,
        graphon: Graphon,
        metrics: Union[List[BaseMetric], BaseMetric],
        progress_bar: bool = True,
        graphon_name: str = "Unnamed",
    ):
        self.graphon = graphon
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.progress_bar = progress_bar
        self.graphon_name = graphon_name
        self.results = {}
        self.graphs = []
        self.generated = False

    def _generate_graphs(self, n_nodes: List[int], n_repetitions: int, random_state: int = 42):
        """Generate graphs from the graphon.

        Will use :py:obj:`self.graphon` to generate :py:obj:`n_repetitions` graphs
        with :py:obj:`n_nodes` nodes and then report the metric values for each
        graph and metrics.

        Parameters:
        ----------
            n_samples: int
                number of nodes in the graph
            n_repetitions: int
                number of times to run the evaluation
        """
        np.random.seed(random_state)
        graphs = []
        for n in n_nodes:
            graphs.append(
                [self.graphon.draw(rho=1.0, n=n, exchangeable=False) for _ in range(n_repetitions)]
            )
        self.graphs = graphs
        self.generated = True

    def evaluate(
        self,
        estimator: BaseEstimator,
        n_nodes: Union[List[int], int],
        n_repetitions: int,
        random_state: int = 42,
    ):
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
        np.random.seed(random_state)
        if isinstance(n_nodes, int):
            n_nodes = [n_nodes]
        estimator_name = estimator.__class__.__name__
        if estimator_name not in self.results:
            self.results[estimator_name] = []

        if not self.generated:
            self._generate_graphs(n_nodes, n_repetitions, random_state)

        results = [np.zeros((len(n_nodes), n_repetitions)) for _ in range(len(self.metrics))]
        for i, _ in enumerate(tqdm(n_nodes, disable=not self.progress_bar)):
            for j in range(n_repetitions):
                graph = self.graphs[i][j]
                estimator.fit(graph)
                for k, metric in enumerate(self.metrics):
                    results[k][i, j] = metric(
                        graphon=self.graphon, estimator=estimator, adj_matrix=graph
                    )
        self.results[estimator_name] = results
        return results

    def __str__(self):
        """Return string representation of the evaluation.

        Returns
        -------
        str
            String representation of the evaluation.
        """
        string = f"Graphon: {self.graphon_name} \n"
        for estimator in self.results:
            string += f"  {estimator} \n"
            for index_metric, metric in enumerate(self.metrics):
                string += f"    {metric}, "
                string += f"mean: {np.mean(self.results[estimator][index_metric]):.4f} "
                string += f"std: {np.std(self.results[estimator][index_metric]):.4f} "
                string += "\n"
        return string
