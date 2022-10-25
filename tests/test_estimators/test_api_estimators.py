"""Test that all the graphon estimators have the API they are supposed to."""

import networkx as nx
import numpy as np
import pytest

from pygraphon.estimators import HistogramEstimator, SimpleMomentEstimator
from pygraphon.graphons import StepGraphon
from pygraphon.utils.utils_matrix import check_symmetric


@pytest.fixture(scope="module")
def sbm_3():
    """Return a 3-block SBM."""
    return nx.stochastic_block_model(
        [10, 10, 10], [[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]], seed=0
    )


class TestApiHistogramEstimator:
    """Test API test for histogram estimator."""

    @pytest.fixture
    def estimator(self):
        hist = HistogramEstimator(bandwithHist=1 / 3)
        return hist

    @pytest.fixture
    def fitted_estimator(self, estimator, sbm_3):
        estimator.fit(sbm_3)
        return estimator

    def test_pij(self, fitted_estimator, sbm_3):
        """Test that the pij estimator returns the pij matrix."""
        P = fitted_estimator.get_edge_connectivity()
        n = sbm_3.number_of_nodes()
        assert P.shape == (n, n)
        assert np.max(P) <= 1
        assert np.min(P) >= 0
        assert check_symmetric(P)

    def test_step_graphon_available(self, fitted_estimator, sbm_3):
        """Test that the step graphon estimated is available."""
        assert isinstance(fitted_estimator.get_graphon(), StepGraphon)


class TestApiMoment(TestApiHistogramEstimator):
    """Test api for Moment estimator."""

    @pytest.fixture
    def estimator(self):
        return SimpleMomentEstimator(blocks=3)

    def test_pij(self, fitted_estimator, sbm_3):
        """Test that the pij estimator is not available."""
        assert fitted_estimator.get_edge_connectivity() is None
