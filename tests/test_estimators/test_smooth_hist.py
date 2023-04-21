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
                np.repeat([number_link_communities - 1], k_init % number_link_communities),
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

    test_array_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_array_1 = estimator._cluster(2, test_array_1)
    assert len(np.unique(labels_array_1)) == 2
    assert (
        (labels_array_1[test_array_1 == 0] == labels_array_1[0])
        & (labels_array_1[test_array_1 == 1] == labels_array_1[4])
    ).all()

    test_array_2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    labels_array_2 = estimator._cluster(3, test_array_2)
    assert len(np.unique(labels_array_2)) == 3
    assert (
        (labels_array_2[test_array_2 == 0] == labels_array_2[0])
        & (labels_array_2[test_array_2 == 1] == labels_array_2[4])
        & (labels_array_2[test_array_2 == 2] == labels_array_2[8])
    ).all()

    test_array_3 = np.array([0.1, 0.15, 0.12, 0.09, 0.23, 0.27, 0.29, 0.3, 0.5, 0.55, 0.48, 0.52])
    labels_array_3 = estimator._cluster(3, test_array_3)
    assert len(np.unique(labels_array_3)) == 3
    assert (
        (labels_array_3[test_array_3 < 0.16] == labels_array_3[0])
        & (labels_array_3[(test_array_3 > 0.2) & (test_array_3 < 0.35)] == labels_array_3[4])
        & (labels_array_3[test_array_3 > 0.45] == labels_array_3[8])
    ).all()


def test_flat_to_tensor() -> None:
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.2], [0.1, 0.2, 0.7]])
    flat_theta = np.array([[0.5, 0.3, 0.1, 0.6, 0.2, 0.7]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    deflatten_theta = estimator._flat_to_tensor(flat_theta, hist)[0]
    assert (deflatten_theta.graphon == hist.graphon).all()  


def test_smoothing_operator() -> None:
    estimator = SmoothNetHist()
    theta = np.array([[1.5, 0.9, 0.3], [0.9, 1.8, 0.6], [0.3, 0.6, 2.1]])
    labels = np.array([0, 1, 1, 0, 1, 0])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)

    assert (theta == hist.graphon).all()

    flat_graphon = hist.graphon[np.triu_indices(3)]
    flat_areas = hist.areas[np.triu_indices(3)]
    computed_smoothed_est = estimator._smoothing_operator(
        tensor_slice_graphon=flat_graphon,
        kmeans=labels,
        number_link_communities=2,
        flat_graphon=flat_graphon,
        flat_areas=flat_areas,
    )
    assert np.allclose(computed_smoothed_est, np.array([[1.8, 0.6, 0.6, 1.8, 0.6, 1.8]]))


def test_smoothing_histLC_givenLC_reg_block() -> None:
    n = 99
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.25], [0.1, 0.25, 0.6]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    adj = hist.draw(rho=None, n=n, exchangeable=False)
    
    smooth_test1 = estimator._smoothing_histLC(hist_approx=hist, A=adj, number_link_communities=3)
    assert np.allclose(
        smooth_test1[-1].graphon, np.array([[1.7, 0.825, 0.3], [0.825, 1.7, 0.825], [0.3, 0.825, 1.7]])
    )

    smooth_test2 = estimator._smoothing_histLC(hist_approx=hist, A=adj, number_link_communities=2)
    assert np.allclose(
        smooth_test2[-1].graphon, np.array([[1.7, 0.65, 0.65], [0.65, 1.7, 0.65], [0.65, 0.65, 1.7]])
    )

    smooth_test3 = estimator._smoothing_histLC(hist_approx=hist, A=adj, number_link_communities=1)
    assert np.allclose(
        smooth_test3[-1].graphon, np.ones((3, 3))
    )


def test_smoothing_histLC_givenLC_nonreg_block() -> None:
    n = 99
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1, 0.7], [0.3, 0.6, 0.25, 0.1], [0.1, 0.25, 0.6, 0.3], [0.7, 0.1, 0.3, 0.01]])
    hist = StepGraphon(theta, bandwidthHist= 0.3)
    adj = hist.draw(rho=None, n=n, exchangeable=False)

    theoretical_theta = np.array([[1.48765248, 0.8288349542857144, 0.27976417, 1.84637743],
                                  [0.8288349542857142, 1.7256768805999998, 0.8288349542857144, 0.2975305],
                                  [0.2975305, 0.8288349542857142, 1.7256768805999998, 0.8288349542857144],
                                  [1.7256768805999998, 0.2975305, 0.8288349542857142, 0.02975305]])
    smooth_test1 = estimator._smoothing_histLC(hist_approx=hist, A=adj, number_link_communities=4)
    #flat_test = estimator._flat_to_tensor(avr_graphon = hist.graphon[np.triu_indices(4)].reshape(1,-1), hist_approx = hist)
    assert np.allclose(smooth_test1[-1].graphon, theoretical_theta)


def test_select_best_graphon() -> None:
    estimator = SmoothNetHist()
    assert estimator._select_best_graphon() == np.inf
