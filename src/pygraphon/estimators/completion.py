"""Implementation of the matrix completion scheme estimator."""
from pygraphon.estimators.BaseEstimator import BaseEstimator


class Completion(BaseEstimator):
    """Estimate graphons via matrix completion scheme."""

    def __init__(
        self,
        rank: int = None,
        tol: float = 1e-3,
        iternumber: int = 20,
        progress: bool = False,
        adjust: bool = True,
    ) -> None:
        self.rank = rank
        self.tol = tol
        self.iternumber = iternumber
        self.progress = progress
        self.adjust = adjust
