"""Smoothing Network Histograms Class"""
from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.HistogramEstimator import HistogramEstimator
from pygraphon.estimators.utils.nethist import nethist
from pygraphon.graphons.StepGraphon import StepGraphon
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
    
    def __init__(self, bandwidth, n_iter=100, n_clusters=10, n_jobs=1, verbose=0, random_state=None):
        super().__init__(bandwidth=bandwidth, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.n_clusters = n_clusters
        self.bandwidth = bandwidth
    
    def _first_fit_hist(self, A):
        """First fit of the histogram estimator."""
        hist = HistogramEstimator(bandwithHist=self.bandwidth)
        hist.fit(graph=A)
        hist_approx = hist.get_graphon()
        return hist_approx
    
    def _smoothing_histLC(self, hist_approx, A):
        """Smoothing of the histogram for link communities."""
        # Extract tiles and reshape
        nrow_tile = hist_approx.graphon.shape[0]
        ncol_tile = hist_approx.graphon.shape[1]
        ntile = int(nrow_tile * ncol_tile)
        P = hist_approx.estimated_edge_connectivity_matrix
        P = P.reshape(ntile, ntile)
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(P)
        labels = kmeans.labels_
        # Average graphon
        avr_graphon = np.zeros((ntile, ntile))
        for i in range(self.n_clusters):
            avr_graphon[labels == i] = np.mean(P[labels == i])
        return avr_graphon, ncol_tile, nrow_tile
    
    def _flatten_graphon(self, avr_graphon, ncol_tile, nrow_tile):
        """Flatten the graphon."""
        # Reshape
        avr_graphon = avr_graphon.reshape(nrow_tile, ncol_tile)
        # Flatten
        avr_graphon = avr_graphon.flatten()
        return avr_graphon
