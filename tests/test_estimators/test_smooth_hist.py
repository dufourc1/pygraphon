import numpy as np
import pytest

from pygraphon.estimators import SmoothNetHist
from pygraphon.graphons import StepGraphon


class DeterministicSmoother(SmoothNetHist):

    def _cluster(self, number_link_communities, flat_graphon) -> np.ndarray:
        k_init = len(np.unique(flat_graphon))
        return np.concatenate(
            [
                np.repeat(np.arange(number_link_communities), k_init // number_link_communities),
                np.repeat([number_link_communities-1], k_init % number_link_communities)
            ]
        )


def test_bic_non_fitted() -> None:
    estimator = SmoothNetHist()
    with pytest.warns():
        assert estimator.get_bic() == np.inf


def test_ratio_par_non_fitted() -> None:
    estimator = SmoothNetHist()
    with pytest.warns():
        assert estimator.get_ratio_par() == np.inf


def test_cluster() -> None:
    estimator = SmoothNetHist()
    assert estimator._cluster(2, np.array([0, 0, 0, 0, 1, 1, 1, 1])) == np.array(
        [0, 0, 0, 0, 1, 1, 1, 1])
    assert estimator._cluster(3, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])) == np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    est_decimals = estimator._cluster(3, np.array(
        [0.1, 0.15, 0.12, 0.09, 0.23, 0.27, 0.29, 0.3, 0.5, 0.55, 0.48, 0.52]))
    assert est_decimals == np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def test_flat_to_tensor() -> None:
    estimator = SmoothNetHist()
    theta = np.array([0.5, 0.3, 0.1], [0.3, 0.6, 0.2], [0.1, 0.2, 0.7])
    flat_theta = np.array([[0.5, 0.3, 0.1, 0.6, 0.2, 0.7]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    deflatten_theta = estimator._flat_to_tensor(flat_theta, hist)[0]
    assert deflatten_theta.graphon == hist.graphon


def test_smoothing_operator() -> None:
    estimator = SmoothNetHist()
    theta = np.array([1.5, 0.9, 0.3], [0.9, 1.8, 0.6], [0.3, 0.6, 2.1])
    labels = np.array([0, 1, 1, 0, 1, 0])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)

    assert theta == hist.graphon

    flat_graphon = hist.areas[np.triu_indices(3)]
    flat_areas = hist.areas[np.triu_indices(3)]
    computed_smoothed_est = estimator._smoothing_operator(tensor_slice_graphon=flat_graphon,
                                                          kmeans=labels,
                                                          number_link_communities=2,
                                                          flat_graphon=flat_graphon,
                                                          flat_areas=flat_areas)
    assert np.allclose(computed_smoothed_est, np.array(
        [[1.8, 0.6, 0.6, 1.8, 0.6, 1.8]]))


def test_select_best_graphon() -> None:
    estimator = SmoothNetHist()
    assert estimator.select_best_graphon() == np.inf
