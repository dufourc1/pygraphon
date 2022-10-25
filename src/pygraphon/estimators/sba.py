"""Implementation of the SBA estimator."""
from pygraphon.estimators.BaseEstimator import BaseEstimator


class SBA(BaseEstimator):
    """Estimate graphon base on SB approximation.

    Parameters
    ----------
    delta : float, optional
        precision parameter larger than 0, by default 0.2
    """

    def __init__(self, delta: float = 0.2) -> None:
        if delta < 0:
            raise ValueError("delta must be larger than 0")
        self.delta = delta
