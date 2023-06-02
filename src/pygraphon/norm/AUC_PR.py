import numpy as np
import sklearn.metrics

from .BaseMetric import ClassificationMetric


class SklearnBinaryMetric(ClassificationMetric):
    """Use a binary classification metric from sklearn.metrics to measure edge predictions.

    Parameters
    ----------
    method : str
        name of the method from `sklearn.metrics`, by default "roc_auc_score".
        Should take these first two arguments: y_true and y_pred (or y_score).
    """

    def __init__(
        self,
        method: str = "roc_auc_score",
        **kwargs,
    ) -> None:
        """Initialise the metric.

        Parameters
        ----------
        method : str
            name of the method from `sklearn.metrics`, by default "roc_auc_score".
            Should take these first two arguments: y_true and y_pred (or y_score).

        Raises
        ------
        ValueError
            if the method is not found in `sklearn.metrics`
        """
        super().__init__()
        try:
            self.method = getattr(sklearn.metrics, method)
        except AttributeError:
            raise ValueError(f"method {method} not found in sklearn.metrics")

    def _compute(
        self,
        adjacency_matrix: np.ndarray,
        pij: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Compute a metric between the probability matrix estimated from an adjacency matrix.

        Parameters
        ----------
        adjacency_matrix: np.ndarray
            adjacency matrix of the graph
        pij: np.ndarray
            probability matrix
        kwargs: dict
            additional arguments to pass to `self.method`

        Returns
        -------
        float
            Binary classification metric on the probability matrix.

        """
        indices = np.triu_indices(adjacency_matrix.shape[0], k=1)
        return self.method(adjacency_matrix[indices], pij[indices], **kwargs)

    def __str__(self) -> str:
        return f"{self.method.__name__} on probability matrix"


class AUCEdge(SklearnBinaryMetric):
    """AUC for edge prediction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(method="roc_auc_score", **kwargs)


class AUPRCEdge(SklearnBinaryMetric):
    """AUPRC for edge prediction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(method="average_precision_score", **kwargs)
