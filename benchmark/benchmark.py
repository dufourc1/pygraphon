from typing import List

from tqdm import tqdm

from pygraphon.estimators import LG, SBA, USVT, HistogramEstimator, NBDsmooth
from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.evaluation.evaluate import Evaluate
from pygraphon.graphons import common_graphons
from pygraphon.norm import BaseMetric, MseProbaEdge


def all_estimators() -> List[BaseEstimator]:
    return [LG(K=3), SBA(), USVT(), HistogramEstimator(), NBDsmooth()]


def all_metrics() -> List[BaseMetric]:
    return [MseProbaEdge()]


if __name__ == "__main__":
    metrics = all_metrics()
    for name, graphon in common_graphons.items():
        evaluator = Evaluate(
            graphon=graphon,
            metrics=all_metrics(),
            progress_bar=False,
            graphon_name=f"graphon_{name}",
        )
        for estimator in tqdm(all_estimators()):
            evaluator.evaluate(estimator=estimator, n_nodes=100, n_repetitions=1)
        print(evaluator)
