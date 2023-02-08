"""Base class for evaluating the performance of an estimator."""
from abc import abstractclassmethod


class Evaluate:
    """Base class for evaluating the performance of an estimator."""

    def __init__(self, estimator, generator, metrics):
        self.estimator = estimator
        self.generator = generator
        self.metrics = metrics

    @abstractclassmethod
    def evaluate(self, n_samples, n_repetitions):
        """Evaluate the performance of the estimator.

        Generate a graph using the generator and estimate the graphon.

        Parameters:
        ----------
            n_samples: int
                number of nodes in the graph
            n_repetitions: int
                number of times to run the evaluation
        """
        # run the estimator n_repetitions times
        # and compute the metrics for each run

    @abstractclassmethod
    def _save_resutls(self, results, path):
        pass
