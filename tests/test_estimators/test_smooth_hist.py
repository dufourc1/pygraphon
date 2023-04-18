import numpy as np
from pygraphon.estimators import SmoothNetHist
import pytest


def test_bic_non_fitted() -> None:
    estimator = SmoothNetHist()
    with pytest.warns():
        assert estimator.get_bic() == np.inf
