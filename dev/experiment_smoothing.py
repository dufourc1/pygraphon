# Import packages
from pygraphon.graphons.StepGraphon import StepGraphon
import numpy as np

import matplotlib.pyplot as plt
from pygraphon.utils.utils_graph import edge_density
from sklearn.cluster import KMeans
from pygraphon.estimators.HistogramEstimator import HistogramEstimator


def error_operator(rel_error, k):
    best_est_id = np.argmin(rel_error)
    best_est_LCk = best_est_id + 1  # gives the number of LC from best estimator
    return (
        rel_error[-1] / rel_error[best_est_id],
        rel_error[best_est_id],
        (int(k * (k + 1) / 2) - best_est_LCk) / int(k * (k + 1) / 2),
        (len(rel_error) - best_est_LCk) / (len(rel_error)),
    )


def smooth_hist(n, rho_n, graphon_matrix, bandwidth):
    A, hist_approx, graphon = initialisation_hist(n, rho_n, graphon_matrix, bandwidth)
    avr_graphon, nclust_hist, nrow_tile = smoothing_histLC(hist_approx, A)
    P_smooth = flat_to_tensor(avr_graphon, nclust_hist, nrow_tile, hist_approx, n)
    smoothed_graphons_error = mse(P_smooth, graphon)
    return graphon, hist_approx, smoothed_graphons_error


def smooth_exp_parallel(i, j, k, n_it, rho_n, theta, temp_folder):
    _, _, rel_error = smooth_hist(n=n_it, rho_n=rho_n, graphon_matrix=theta, bandwidth=1 / k)
    metric_result = error_operator(rel_error, k)
    # save metric list
    np.save(f"temp/{temp_folder}metric_list_{i}_{j}_{k}_{n_it}.npy", np.array(metric_result))
    return metric_result


# ------------Initialisation----------------


def initialisation_hist(n, rho_n, graphon_matrix, bandwidth):
    # set up normalised graphon with graphon
    graphon = StepGraphon(graphon_matrix, bandwidth)
    # Set Adjacency
    A = graphon.draw(rho=rho_n, n=n, exchangeable=False)

    # Apply histogram anc connectivity matrix
    hist = HistogramEstimator(bandwithHist=bandwidth)
    hist.fit(graph=A)
    hist_approx = hist.get_graphon()
    # P = hist.estimated_edge_connectivity_matrix
    return A, hist_approx, graphon


# ------------Smoothing of Hist for Link Communities-------------


def smoothing_histLC(hist_approx, A):
    # Extract tiles and reshape
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


# -----------Reshaping Data into tensor-----------


def flat_to_tensor(avr_graphon, nclust_hist, nrow_tile, hist_approx, n):
    # Reverse flattening back to 2d
    # initiate matrix containing smooth graphon
    KM_smooth_g = np.zeros((nclust_hist, nrow_tile, nrow_tile))  # max_k?
    i, j = np.triu_indices(nrow_tile)
    KM_smooth_g[:, i, j] = avr_graphon
    # Make it symmetric by extracting diag, making it 3d and doing X+X^T-diag(X)
    # Extract diagonal
    i, j = np.diag_indices(nrow_tile)
    diag_km_vec = np.diagonal(KM_smooth_g, axis1=1, axis2=2)
    # make 3d array with diagonal in axis 1&2
    diag_km = np.zeros((nclust_hist, nrow_tile, nrow_tile))  # max_k ?
    diag_km[:, i, j] = diag_km_vec
    # X+X^T-diag(X) on axis (1,2) of tensor
    sym_KM_smooth_g = KM_smooth_g + np.transpose(KM_smooth_g, (0, 2, 1)) - diag_km
    # Get smooth graphon into correct form and \hat{pij} for the whole tensor
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


# ---------------Evaluate Error--------------


def mse(P_smooth, graphon):
    # Compute error, MSE
    smoothed_graphons_error = np.zeros(P_smooth.shape[0])
    P_star = graphon._get_edge_probabilities(P_smooth[0].shape[0], False, True)  # theoretical P
    diag_eye = np.eye(P_star.shape[0])
    for i in range(P_smooth.shape[0]):
        # add diag of ones for division -> does not matter
        smoothed_graphons_error[i] = np.linalg.norm((P_star - P_smooth[i])) ** 2 / (
            P_smooth.shape[1] ** 2
        )
    return smoothed_graphons_error


# ------------------PLOT-------------------


def smooth_heatmap(n_list, k_list, error, title):
    x, y = np.meshgrid(n_list, k_list)
    fig, ax = plt.subplots()

    c = ax.imshow(
        error.T.T,
        cmap="Reds",
        interpolation="spline36",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        vmin=np.min(error),
        vmax=np.max(error),
        origin="lower",
    )
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()
    return fig, ax
    # antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    # 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
    # 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman', 'jet'


def smooth_heatmap2(n_list, k_list, error, main_title, subtitle, tile_type):
    k_list = np.array(k_list)
    fig, ax = plt.subplots(2, 2)
    k = 0
    for j in range(2):
        for i in range(2):
            c = ax[j, i].imshow(
                error[k],
                cmap="Reds",
                interpolation=tile_type,
                extent=[n_list.min(), n_list.max(), k_list.min(), k_list.max()],
                aspect="auto",
                vmin=np.min(error[k]),
                vmax=np.max(error[k]),
                origin="lower",
            )
            ax[j, i].set_title(subtitle[k])
            # set the limits of the plot to the limits of the data
            ax[j, i].axis([n_list.min(), n_list.max(), k_list.min(), k_list.max()])
            plt.colorbar(c, ax=ax[j, i])
            k += 1
    # set labels
    plt.setp(ax[-1, :], xlabel="n")
    plt.setp(ax[:, 0], ylabel="k")
    fig.suptitle(main_title)
    plt.tight_layout()
    plt.show()
    return fig, ax
