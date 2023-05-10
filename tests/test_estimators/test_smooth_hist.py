import numpy as np
import pytest

from pygraphon.estimators import HistogramEstimator, SmoothNetHist
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


def _test_cluster_up_to_relabelling(theoretical: np.ndarray, empirical: np.ndarray):
    """Test that two sets of labels are the up to relabelling."""
    assert np.array_equal(np.unique(empirical), np.unique(theoretical))
    assert empirical.shape == theoretical.shape
    for g in np.unique(theoretical):
        # find corresponding label
        ids_true = np.where(theoretical == g)[0]
        ids_pred = np.where(empirical == empirical[ids_true[0]])[0]
        assert np.array_equal(ids_true, ids_pred)


def test_non_fitted_getters() -> None:
    """Test non fitted estimator returns default value."""
    estimator = SmoothNetHist()
    with pytest.warns():
        assert estimator.get_bic() == np.inf
    with pytest.warns():
        assert estimator.get_ratio_par() == np.inf


def test_cluster_no_reduction() -> None:
    """Test the clustering method for basic cases.

    Should return the same group up to relabelling if there are no reductions
    in the number of groups.
    """
    estimator = SmoothNetHist()

    test_array_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_array_1 = estimator._cluster(2, test_array_1)
    _test_cluster_up_to_relabelling(test_array_1, labels_array_1)

    test_array_2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    labels_array_2 = estimator._cluster(3, test_array_2)
    _test_cluster_up_to_relabelling(test_array_2, labels_array_2)


def test_cluster_reduction() -> None:
    """Test if the clustering method reduces the number of groups."""
    estimator = SmoothNetHist()
    test_array_3 = np.array([0.1, 0.15, 0.12, 0.09, 0.23, 0.27, 0.29, 0.3, 0.5, 0.55, 0.48, 0.52])
    labels_array_3 = estimator._cluster(3, test_array_3)
    assert len(np.unique(labels_array_3)) == 3
    assert test_array_3.shape == labels_array_3.shape


def test_flat_to_tensor() -> None:
    """Test flat_to_tensor with regular block graphon."""
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.2], [0.1, 0.2, 0.7]])
    flat_theta = np.array([[0.5, 0.3, 0.1, 0.6, 0.2, 0.7]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    deflatten_theta = estimator._flat_to_tensor(flat_theta, hist)[0]
    assert (deflatten_theta.graphon == hist.graphon).all()


def test_smoothing_operator() -> None:
    """Test smoothing_operator with regular block graphon."""
    estimator = SmoothNetHist()
    theta = np.array([[1.5, 0.9, 0.3], [0.9, 1.8, 0.6], [0.3, 0.6, 2.1]])
    labels = np.array([0, 1, 1, 0, 1, 0])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)

    assert (theta == hist.graphon).all()

    flat_graphon = hist.graphon[np.triu_indices(3)]
    flat_areas = hist.areas[np.triu_indices(3)]
    computed_smoothed_est = estimator._smoothing_operator(
        tensor_slice_graphon=flat_graphon,
        labels=labels,
        number_link_communities=2,
        flat_graphon=flat_graphon,
        flat_areas=flat_areas,
    )
    assert np.allclose(computed_smoothed_est, np.array([[1.8, 0.6, 0.6, 1.8, 0.6, 1.8]]))


def test_smoothing_histLC_givenLC_reg_block() -> None:
    """Test smoothing_histLC_givenLC with regular block graphon."""
    n = 99
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.25], [0.1, 0.25, 0.6]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    adj = hist.draw(rho=None, n=n, exchangeable=False)

    smooth_test1, n_link_com = estimator._smoothing_histLC(
        hist_approx=hist, A=adj, number_link_communities=3
    )
    assert np.allclose(
        smooth_test1[-1].graphon,
        np.array([[1.7, 0.825, 0.3], [0.825, 1.7, 0.825], [0.3, 0.825, 1.7]]),
    )
    assert np.array_equal(n_link_com, [3])

    smooth_test2, n_link_com = estimator._smoothing_histLC(
        hist_approx=hist, A=adj, number_link_communities=2
    )
    assert np.allclose(
        smooth_test2[-1].graphon,
        np.array([[1.7, 0.65, 0.65], [0.65, 1.7, 0.65], [0.65, 0.65, 1.7]]),
    )
    assert np.array_equal(n_link_com, [2])

    smooth_test3, n_link_com = estimator._smoothing_histLC(
        hist_approx=hist, A=adj, number_link_communities=1
    )
    assert np.allclose(smooth_test3[-1].graphon, np.ones((3, 3)))
    assert np.array_equal(n_link_com, [1])


def test_smoothing_histLC_givenLC_nonreg_block() -> None:
    """Test smoothing_histLC_givenLC with non-regular block graphon."""
    n = 99
    estimator = SmoothNetHist()
    theta = np.array(
        [[0.5, 0.3, 0.1, 0.7], [0.3, 0.6, 0.25, 0.1], [0.1, 0.25, 0.6, 0.3], [0.7, 0.1, 0.3, 0.01]]
    )
    hist = StepGraphon(theta, bandwidthHist=0.3)
    adj = hist.draw(rho=None, n=n, exchangeable=False)

    theoretical_theta = np.array(
        [
            [1.48765248, 0.82883496, 0.27693223, 1.82768734],
            [0.82883496, 1.82768734, 0.82883496, 0.27693223],
            [0.27693223, 0.82883496, 1.82768734, 0.82883496],
            [1.82768734, 0.27693223, 0.82883496, 0.27693223],
        ]
    )
    estimator._num_par_nethist = len(np.unique(hist.graphon))
    smooth_test1, n_link_com = estimator._smoothing_histLC(
        hist_approx=hist, A=adj, number_link_communities=4
    )
    assert np.allclose(smooth_test1[-1].graphon * smooth_test1[-1].initial_rho, theoretical_theta)
    assert np.array_equal(n_link_com, [4])


def test_smoothing_histLC_tensor() -> None:
    """Test smoothing_histLC with regular block graphon, selected by BIC."""
    n = 99
    estimator = SmoothNetHist()
    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.25], [0.1, 0.25, 0.6]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    adj = hist.draw(rho=None, n=n, exchangeable=False)

    olhede_fit, _ = estimator._first_approximate_graphon_from_adjacency(adjacencyMatrix=adj)
    params_olhede_fit = (olhede_fit.graphon.shape[0] + 1) * olhede_fit.graphon.shape[0] // 2
    smooth_test_tensor, n_link_com = estimator._smoothing_histLC(
        hist_approx=olhede_fit, A=adj, number_link_communities=None
    )
    assert np.array_equal(n_link_com, np.arange(1, params_olhede_fit + 1))

    assert estimator._num_par_nethist == len(smooth_test_tensor)
    assert np.allclose(smooth_test_tensor[-1].graphon, olhede_fit.graphon)
    assert np.allclose(smooth_test_tensor[0].graphon, np.ones(smooth_test_tensor[0].graphon.shape))


def test_deterministic_clustering() -> None:
    """Test end to end pipeline without kmeans randomness."""
    estimator = DeterministicSmoother()
    nethist_estimator = HistogramEstimator()

    theta = np.array([[0.5, 0.3, 0.1], [0.3, 0.6, 0.25], [0.1, 0.25, 0.6]])
    hist = StepGraphon(theta, bandwidthHist=1 / 3)
    n = 100

    adj = hist.draw(rho=None, n=n, exchangeable=False)

    nethist_estimator.fit(graph=adj)
    estimator.fit(graph=adj)

    K = nethist_estimator.get_graphon().graphon.shape[0]
    assert estimator._num_par_nethist == (K * (K - 1)) // 2 + K
    assert estimator._num_par_smooth == len(np.unique(estimator.get_graphon().graphon))
    test_clustered, n_link_coms = estimator._smoothing_histLC(
        nethist_estimator.get_graphon(), adj, number_link_communities=estimator._num_par_smooth
    )
    assert np.array_equal(n_link_coms, [estimator._num_par_smooth])
    assert np.array_equiv(test_clustered[-1].graphon, estimator.get_graphon().graphon)
