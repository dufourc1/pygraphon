from typing import List

from loguru import logger
from tqdm import tqdm

from pygraphon.estimators import LG, SBA, USVT, HistogramEstimator, NBDsmooth
from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.evaluation.evaluate import Evaluate
from pygraphon.graphons import common_graphons
from pygraphon.norm import BaseMetric, MseProbaEdge


def all_estimators() -> List[BaseEstimator]:
    """Return a list of all estimators to evaluate.

    Returns
    -------
    List[BaseEstimator]
        List of all estimators to evaluate.
    """
    return [LG(K=3), SBA(), USVT(), HistogramEstimator(), NBDsmooth()]


def all_metrics() -> List[BaseMetric]:
    """Return a list of all metrics to evaluate.

    Returns
    -------
    List[BaseMetric]
        List of all metrics to evaluate.
    """
    return [MseProbaEdge()]


def benchmark():
    """Run and print the benchmark."""
    for name, graphon in common_graphons.items():
        evaluator = Evaluate(
            graphon=graphon,
            metrics=all_metrics(),
            progress_bar=False,
            graphon_name=f"graphon_{name}",
        )
        for estimator in tqdm(all_estimators()):
            evaluator.evaluate(estimator=estimator, n_nodes=100, n_repetitions=1)
        logger.info(evaluator)


if __name__ == "__main__":
    benchmark()
