"""Smoothing Network Histograms Class"""
from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.HistogramEstimator import HistogramEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.estimators.networkhistogram.nethist  import nethist
from pygraphon.utils.utils_graph import edge_density

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class SmoothNetHist(BaseEstimator):
    """Implementation of the smoothing network histograms algorithm.

    From the theory of edge clustering, approximate a graphon by smoothing blocks of similar density from a histogram.

    Args:
        bandwidth (float): bandwidth of the graphon.
        n_iter (int): number of iterations.
        n_clusters (int): number of clusters.
        n_jobs (int): number of jobs to run in parallel.
        verbose (int): verbosity level.
        random_state (int): random state.
    """
    
    def __init__(self, bandwidth, n_list, k_list):
        super().__init__(bandwidth=bandwidth)
        self.bandwidth = bandwidth
        self.n_list = n_list
        self.k_list = k_list
    
    def _first_fit_hist(self, A):
        """First fit of the histogram estimator."""
        hist = HistogramEstimator(bandwithHist=self.bandwidth, method="mine")
        hist.fit(graph=A)
        hist_approx = hist.get_graphon()
        return hist_approx
    
    def _smoothing_histLC(self, hist_approx, A):
        """Smoothing of the histogram estimator."""
        nrow_tile = hist_approx.graphon.shape[0]
        flat_graphon = hist_approx.graphon[np.triu_indices(nrow_tile)]
        # Clustering according to kmeans
        nclust_hist = len(np.unique(flat_graphon))
        # copies for different number of clusters
        avr_graphon = np.asarray([flat_graphon.copy() for _ in range(nclust_hist)])
        for i in range(nclust_hist - 1):
            if i == 0:
                avr_graphon[i] = edge_density(A) * np.ones_like(flat_graphon)
            else:
                kmeans = KMeans(n_clusters=i + 1, random_state=0, init="k-means++", n_init=10).fit(
                    flat_graphon.reshape(-1, 1)
                )
                # Smooth using mean by memberhsip
                for j in range(i + 1):  # average tiles of same community
                    avr_graphon[i][kmeans.labels_ == j] = flat_graphon[kmeans.labels_ == j].mean()
        return avr_graphon, nclust_hist, nrow_tile
    
    def _flat_to_tensor(avr_graphon, nclust_hist, nrow_tile, hist_approx, n):
        """Convert the smoothed histogram to a tensor."""
        KM_smooth_g = np.zeros((nclust_hist, nrow_tile, nrow_tile))  # max_k?
        i, j = np.triu_indices(nrow_tile)
        KM_smooth_g[:, i, j] = avr_graphon

        # Make it symmetric 
        i, j = np.diag_indices(nrow_tile)
        diag_km_vec = np.diagonal(KM_smooth_g, axis1=1, axis2=2)
        # make 3d array with diagonal in axis 1&2
        diag_km = np.zeros((nclust_hist, nrow_tile, nrow_tile))  # max_k ?
        diag_km[:, i, j] = diag_km_vec
        # X+X^T-diag(X) on axis (1,2) of tensor
        sym_KM_smooth_g = KM_smooth_g + np.transpose(KM_smooth_g, (0, 2, 1)) - diag_km

        smoothed_graphons = [
            StepGraphon(
                graphon=sym_KM_smooth_g[i],
                bandwidthHist=hist_approx.bandwidthHist,
                initial_rho=hist_approx.initial_rho,
            )
            for i in range(sym_KM_smooth_g.shape[0])
        ]
        P_smooth = np.array(
            [g._get_edge_probabilities(n=n, exchangeable=False) for g in smoothed_graphons]
        )
        return P_smooth
