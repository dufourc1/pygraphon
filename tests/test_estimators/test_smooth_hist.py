import numpy as np
import pytest

from pygraphon.estimators import SmoothNetHist


def test_bic_non_fitted() -> None:
    estimator = SmoothNetHist()
    with pytest.warns():
        assert estimator.get_bic() == np.inf
