# Import packages
from pygraphon.graphons.StepGraphon import StepGraphon
import numpy as np
from pygraphon.plots.plot_matrices import spy
import matplotlib.pyplot as plt
from pygraphon.utils.utils_graph import edge_density
from sklearn.cluster import KMeans
from pygraphon.estimators.HistogramEstimator import HistogramEstimator
from pygraphon.estimators.MomentEstimator import MomentEstimator, SimpleMomentEstimator


def smooth_hist(n, rho_n, graphon_matrix, bandwidth):

    # ------------Initialisation----------------

    # set up normalised graphon with graphon
    graphon = StepGraphon(graphon_matrix, bandwidth)
    # Set Adjacency
    A = graphon.draw(rho=rho_n, n=n, exchangeable=False)

    # Apply histogram anc connectivity matrix
    hist = HistogramEstimator()
    hist_approx = hist.estimate(adjacency_matrix=A, bandwidthHist=None)
    #P = hist.estimated_edge_connectivity_matrix

    # ------------Smoothing of Hist for Link Communities-------------

    # Extract tiles and reshape
    nrow_tile = hist_approx.graphon.shape[0]
    flat_graphon = hist_approx.graphon[np.triu_indices(nrow_tile)]
    # Clustering according to kmeans
    nclust_detect_hist = len(np.unique(flat_graphon))
    # max_k = int(nrow_tile*(nrow_tile+1)/2)  # number of individual tile
    # copies for different number of clusters
    avr_graphon = np.asarray([flat_graphon.copy()
                             for k in range(nclust_detect_hist)])
    #print(f"Hist returned {nrow_tile} communities, hence {max_k} tiles")
    #print(f"flat_graphon {flat_graphon}")
    # print(len(np.unique(flat_graphon)))
    for i in range(nclust_detect_hist-1):
        if i == 0:
            avr_graphon[i] = edge_density(A)*np.ones_like(flat_graphon)
        else:
            kmeans = KMeans(n_clusters=i+1, random_state=0, init="k-means++",
                            n_init=10).fit(flat_graphon.reshape(-1, 1))
            #print(i+1, np.unique(kmeans.labels_, return_counts=True))
            # Smooth using mean by memberhsip
            for j in range(i+1):  # average tiles of same community
                avr_graphon[i][kmeans.labels_ ==
                               j] = flat_graphon[kmeans.labels_ == j].mean()

    # -----------Reshaping Data into tensor-----------

    # Reverse flattening back to 2d
    # initiate matrix containing smooth graphon
    KM_smooth_graphon = np.zeros(
        (nclust_detect_hist, nrow_tile, nrow_tile))  # max_k?
    i, j = np.triu_indices(nrow_tile)
    KM_smooth_graphon[:, i, j] = avr_graphon
    # Make it symmetric by extracting diag, making it 3d and doing X+X^T-diag(X)
    # Extract diagonal
    i, j = np.diag_indices(nrow_tile)
    diag_KM_vec = np.diagonal(KM_smooth_graphon, axis1=1, axis2=2)
    # make 3d array with diagonal in axis 1&2
    diag_km = np.zeros((nclust_detect_hist, nrow_tile, nrow_tile))  # max_k ?
    diag_km[:, i, j] = diag_KM_vec
    # X+X^T-diag(X) on axis (1,2) of tensor
    sym_KM_smooth_graphon = KM_smooth_graphon + \
        np.transpose(KM_smooth_graphon, (0, 2, 1))-diag_km
    # Get smooth graphon into correct form and \hat{pij} for the whole tensor
    smoothed_graphons = [StepGraphon(graphon=sym_KM_smooth_graphon[i], bandwidthHist=hist_approx.bandwidthHist,
                                     initial_rho=hist_approx.initial_rho) for i in range(sym_KM_smooth_graphon.shape[0])]
    P_smooth = np.array([g._get_edge_probabilities(
        n=n, exchangeable=False) for g in smoothed_graphons])

    # ---------------Evaluate Error--------------

    # Compute error, MSE
    smoothed_graphons_error = np.zeros(P_smooth.shape[0])
    P_star = graphon._get_edge_probabilities(
        P_smooth[0].shape[0], False, True)  # theoretical P
    diag_eye = np.eye(P_star.shape[0])
    for i in range(P_smooth.shape[0]):
        smoothed_graphons_error[i] = np.linalg.norm(
            (P_star-P_smooth[i])/(diag_eye + P_star))/(n*(n-1)/2)  # add diag of ones for division -> does not matter
    # Store P estimated by Hist and Best P from smoothing (may be P from Hist)
    P_hist = hist_approx._get_edge_probabilities(n, False, True)
    P_best_est_smooth = P_smooth[np.argmin(smoothed_graphons_error)]

    return graphon, P_hist, P_best_est_smooth, smoothed_graphons_error


def smooth_heatmap(n_list, k_list, error, title):
    x, y = np.meshgrid(n_list, k_list)
    fig, ax = plt.subplots()

    c = ax.imshow(error.T.T, cmap='Reds', interpolation='spline36',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  aspect="auto",
                  vmin=np.min(error),
                  vmax=np.max(error),
                  origin="lower"
                  )
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()
    return fig, ax
    # antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 
    # 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 
    # 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
def smooth_heatmap2(n_list, k_list, error, title):
    #x, y = np.meshgrid(n_list, k_list)
    k_list = np.array(k_list)
    fig, ax = plt.subplots(2,2)
    k=0
    for j in range(2):
        for i in range(2):
            c = ax[j,i].imshow(error[k], cmap='Reds', interpolation='spline36',
                        extent=[n_list.min(), n_list.max(), k_list.min(), k_list.max()],
                        aspect="auto",
                        vmin=np.min(error[k]),
                        vmax=np.max(error[k]),
                        origin="lower"
                        )
            ax[j,i].set_title(title[k])
            # set the limits of the plot to the limits of the data
            ax[j,i].axis([n_list.min(), n_list.max(), k_list.min(), k_list.max()])
            plt.colorbar(c, ax=ax[j,i])
            k+=1
            print(k)
    plt.show()
    return fig, ax