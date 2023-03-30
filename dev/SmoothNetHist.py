"""Smoothing Network Histograms Class"""
from typing import Optional, Tuple

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.estimators.networkhistogram.HistogramEstimator import HistogramEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_graph import edge_density

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans


class SmoothNetHist(BaseEstimator):
    """Implementation of the smoothing network histograms algorithm.

    From the theory of edge clustering, approximate a graphon by smoothing blocks of similar density from a histogram.

    Args:
        bandwidth (float): bandwidth of the graphon.
    """
    
    def __init__(self, bandwidth: Optional[float]= None, method: str = "mine") -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.method = method

    def _first_fit_hist(
            self,
            adjacency_matrix: np.ndarray,
            bandwidthHist: Optional[float] = None,
            ):
        """First fit of the histogram estimator.
        
        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjancency matrix of the graph
        bandwidthHist : float
            size of the blocks (between 0 and 1), by default None

        Returns
        -------
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator, not matrix of connection Pij of size n x n
        """
        if bandwidthHist is None:
            bandwidthHist = self.bandwidth
        hist = HistogramEstimator(bandwithHist=bandwidthHist, method=self.method)
        hist.fit(graph=adjacency_matrix)
        return hist
    
    def _smoothing_histLC(self, 
                          hist_approx : HistogramEstimator,
                          A : np.ndarray
                          ):
        """Smoothing of the histogram estimator using k-means++ [1].
        
        Parameters
        ----------
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator
        A : np.ndarray
            adjancency matrix of the graph
            
        Returns
        -------
        avr_graphon : np.ndarray
            list of kmean-smoothed graphon in increasing order of number of link communities,
            avr_graphon[i] is the smoothed graphon with i+1 link communities.
        
        References
        ----------
        [1] Arthur, D. and Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding.
        """
        nrow_tile = hist_approx.graphon.shape[0]
        flat_graphon = hist_approx.graphon[np.triu_indices(nrow_tile)]

        # Clustering according to kmeans
        nclust_hist = len(np.unique(flat_graphon))
        avr_graphon = np.asarray([flat_graphon.copy() for _ in range(nclust_hist)])
        for i in range(nclust_hist - 1):
            if i == 0:
                avr_graphon[i] = edge_density(A) * np.ones_like(flat_graphon)
            else:
                kmeans = KMeans(n_clusters=i + 1, random_state=0, init="k-means++", n_init=10).fit(
                    flat_graphon.reshape(-1, 1)
                )
                for j in range(i + 1):  # average tiles of same community
                    avr_graphon[i][kmeans.labels_ == j] = flat_graphon[kmeans.labels_ == j].mean()

        return avr_graphon
    
    def _flat_to_tensor(self,
                        avr_graphon : np.ndarray,
                        hist_approx : StepGraphon,
                        n : int
                        ):
        """Convert the flattened smoothed histograms to a tensor containing all possible smoothing.
        
        Parameters
        ----------
        avr_graphon : np.ndarray
            list of smoothed graphon in increasing order of number of link communities,
        hist_approx : StepGraphon
            approximated graphon from the histogram estimator
        n : int
            number of nodes of the graph
        
        Returns
        -------
        P_smooth : np.ndarray
            matrix of connection Pij of size n x n for each smoothed graphon
        """
        nrow_tile = hist_approx.graphon.shape[0]
        nclust_hist = avr_graphon.shape[0]
        #deflatte the smoothed histograms and put them in a tensor
        KM_smooth_g = np.zeros((nclust_hist, nrow_tile, nrow_tile))
        i, j = np.triu_indices(nrow_tile)
        KM_smooth_g[:, i, j] = avr_graphon

        # Make it symmetric 
        i, j = np.diag_indices(nrow_tile)
        diag_km_vec = np.diagonal(KM_smooth_g, axis1=1, axis2=2)
        diag_km = np.zeros((nclust_hist, nrow_tile, nrow_tile))
        diag_km[:, i, j] = diag_km_vec
        # X+X^T-diag(X) on axis (1,2) of tensor
        sym_KM_smooth_g = KM_smooth_g + np.transpose(KM_smooth_g, (0, 2, 1)) - diag_km

        smoothed_graphons = [
            StepGraphon(
                graphon=sym_KM_smooth_g[i],
                bandwidthHist=hist_approx.bandwidthHist,
                initial_rho=hist_approx.initial_rho,
            )
            for i in range(nclust_hist)
        ]
        P_smooth = np.array(
            [g._get_edge_probabilities(n=n, exchangeable=False) for g in smoothed_graphons]
        )
        return P_smooth
