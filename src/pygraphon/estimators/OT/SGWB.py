"""Implementation of empirical degrees based estimator."""

import copy
from typing import Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import ot

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import StepGraphon


class SGWB(BaseEstimator):
    """Larget Gap algorithm :cite:p:`channarond2012`.

    Parameters
    ----------
    smoothed : bool, optional
        If `True`, estimate the graphon with a smoothed Gromov-Wasserstein barycenter.
    gamma: Float, optional
        trade off parameter for the fused Gromov-Wasserstein barycenter. Should be between 0 and 1.
    epsilon: Float, optional
        regularization parameter for the sinkhorn algorithm.
    alpha: Float, optional
        regularization parameter for the smoothness penalty of the Gromov-Wasserstein barycenter.
    """

    def __init__(
        self, smoothed: bool = False, gamma: float = 1, epsilon: float = 5e-3, alpha: float = 3e-4
    ) -> None:
        super().__init__()
        self.smoothed = smoothed
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.inner_iters = 20
        self.outer_iters = 50
        self.num_blocks = lambda x: int(x / int(np.log2(x) + 1))

    def fit_multiple(
        self, graphs: Union[Iterable[nx.Graph], Iterable[np.ndarray]], *args, **kwargs
    ):
        """Estimate the graphon function f(x,y) values from multiple realized graphs or adjacency matrices.

        Parameters
        ----------
        graphs : Union[Iterable[nx.Graph], Iterable[np.ndarray]]
            Iterable of networkx simple graphs or adjacency matrices
        args: optional
            additional arguments
        kwargs: optional
            additional arguments

        Raises
        ------
        ValueError
            type of graph is not supported
        ValueError
            if the graph is not a simple graph
        """

        self.fitted = True
        if isinstance(graphs[0], nx.Graph):
            graphs = [nx.to_numpy_array(graph) for graph in graphs]

        aligned_graphs, ps, max_num, min_num = align_graphs(graphs, padding=False)
        p = estimate_target_distribution(ps, self.num_blocks(max_num))
        ws = np.ones(len(aligned_graphs)) / len(aligned_graphs)

        if self.smoothed:
            self.maps, stepfunc = smoothed_fgw_barycenter(
                aligned_graphs,
                ps,
                p,
                ws,
                alpha=self.alpha,
                inner_iters=self.inner_iters,
                outer_iters=self.outer_iters,
                beta=self.epsilon,
                gamma=self.gamma,
            )
        else:
            self.maps, stepfunc = fgw_barycenter(
                aligned_graphs,
                ps,
                p,
                ws,
                inner_iters=self.inner_iters,
                outer_iters=self.outer_iters,
                beta=self.epsilon,
                gamma=self.gamma,
                which="original",
            )
        self.graphon = StepGraphon(stepfunc, bandwidthHist=1 / stepfunc.shape[0])

    def _approximate_graphon_from_adjacency(
        self,
        adjacency_matrix: np.ndarray,
    ) -> Tuple[StepGraphon, np.ndarray]:
        self.fit_multiple([adjacency_matrix])
        return self.graphon, None


def smoothed_fgw_barycenter(
    aligned_graphs: List[np.ndarray],
    aligned_ps: List[np.ndarray],
    p_b: np.ndarray,
    ws: np.ndarray,
    alpha: float,
    inner_iters: int,
    outer_iters: int,
    beta: float,
    gamma: float,
    **kwargs,
) -> np.ndarray:
    """
    Calculate smoothed Gromov-Wasserstein barycenter

    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param aligned_ps: a list of (Ni, 1) distributions
    :param p_b: (Nb, 1) distribution
    :param ws: (K, ) weights
    :param alpha: the weight of smoothness regularizer
    :param inner_iters: the number of sinkhorn iterations
    :param outer_iters: the number of barycenter iterations
    :param beta: the weight of proximal term
    :param gamma: the weight of gw term
    :return:
    """
    nb = p_b.shape[0]
    dmat = (
        2 * np.concatenate((np.zeros((nb - 2, 1)), np.eye(nb - 2), np.zeros((nb - 2, 1))), axis=1)
        - np.concatenate((np.zeros((nb - 2, 2)), np.eye(nb - 2)), axis=1)
        - np.concatenate((np.eye(nb - 2), np.zeros((nb - 2, 2))), axis=1)
    )  # (nb - 2, nb)

    lmat = alpha * (dmat.T @ dmat)
    us, ss, _ = np.linalg.svd(lmat)

    numerator = ss**2 + p_b[:, 0] ** 2  # + 1e-16
    lmat2 = us @ np.diag(ss / numerator) @ us.T
    pmat2 = np.diag(p_b[:, 0] / numerator)

    cost_ps = []
    trans = []
    priors = []
    for p in aligned_ps:
        cost_p = p_b**2 + (p**2).T - 2 * (p_b @ p.T)
        cost_p /= np.max(cost_p)
        cost_ps.append(cost_p)
        tran = proximal_ot(cost_p, p_b, p, iters=inner_iters, beta=beta)
        trans.append(tran)
        priors.append(tran + 1e-16)

    barycenter = None
    for o in range(outer_iters):
        # update smoothed barycenter
        averaged_graph = averaging_graphs(aligned_graphs, trans, ws)
        barycenter = lmat2 @ averaged_graph @ lmat2.T + pmat2 @ averaged_graph @ pmat2.T
        # update optimal transports
        for i in range(len(aligned_graphs)):
            cost_i = gw_cost(barycenter, aligned_graphs[i], trans[i], p_b, aligned_ps[i])
            cost_i = gamma * cost_i + (1 - gamma) * cost_ps[i]
            cost_i /= np.max(cost_i)
            trans[i] = proximal_ot(
                cost_i, p_b, aligned_ps[i], iters=inner_iters, beta=beta, prior=priors[i]
            )

    barycenter[barycenter > 1] = 1
    barycenter[barycenter < 0] = 0
    return barycenter, trans


def fgw_barycenter(
    aligned_graphs: List[np.ndarray],
    aligned_ps: List[np.ndarray],
    p_b: np.ndarray,
    ws: np.ndarray,
    inner_iters: int,
    outer_iters: int,
    beta: float,
    gamma: float,
    which="original",
    **kwargs,
) -> np.ndarray:
    """
    Calculate smoothed Gromov-Wasserstein barycenter

    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param aligned_ps: a list of (Ni, 1) distributions
    :param p_b: (Nb, 1) distribution
    :param ws: (K, ) weights
    :param inner_iters: the number of sinkhorn iterations
    :param outer_iters: the number of barycenter iterations
    :param beta: the weight of proximal term
    :param gamma: the weight of gw term
    :return:
    """
    cost_ps = []
    trans = []
    priors = []
    for p in aligned_ps:
        cost_p = p_b**2 + (p**2).T - 2 * (p_b @ p.T)
        cost_p /= np.max(cost_p)
        cost_ps.append(cost_p)
        tran = proximal_ot(cost_p, p_b, p, iters=inner_iters, beta=beta)
        trans.append(tran)
        priors.append(tran + 1e-16)

    barycenter = None
    for o in range(outer_iters):
        # update smoothed barycenter
        averaged_graph = averaging_graphs(aligned_graphs, trans, ws)
        barycenter = averaged_graph / (p_b @ p_b.T)

        # update optimal transports
        for i in range(len(aligned_graphs)):
            cost_i = gw_cost(barycenter, aligned_graphs[i], trans[i], p_b, aligned_ps[i])
            cost_i = gamma * cost_i + (1 - gamma) * cost_ps[i]
            cost_i /= np.max(cost_i)
            if which == "original":
                trans[i] = proximal_ot(
                    cost_i, p_b, aligned_ps[i], iters=inner_iters, beta=beta, prior=priors[i]
                )
            elif which == "ot":
                trans[i] = ot.sinkhorn(
                    p_b.reshape(-1), aligned_ps[i].reshape(-1), cost_i, 5e-3, numItermax=inner_iters
                )
            elif which == "smooth":
                trans[i] = ot.smooth.smooth_ot_dual(
                    p_b.reshape(-1), aligned_ps[i].reshape(-1), cost_i, 5e-3
                )
            else:
                raise ValueError("which should be one of 'original', 'ot', 'smooth'")

    # barycenter[barycenter > 1] = 1
    # barycenter[barycenter < 0] = 0
    return barycenter, trans


def averaging_graphs(
    aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray
) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(
    cost: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    iters: int,
    beta: float,
    error_bound: float = 1e-10,
    prior: np.ndarray = None,
) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans


def node_cost_st(
    cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray
) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s**2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t**2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(
    cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray
) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)


def estimate_target_distribution(probs: List[np.ndarray], dim_t: int = None) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a list of node distributions [(n_s, 1) the distribution of source nodes]
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    if dim_t is None:
        dim_t = min([probs[i].shape[0] for i in range(len(probs))])

    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for i in range(len(probs)):
        p_s = probs[i][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_i = np.interp(x_t, x_s, p_s) + 1e-3
        p_t[:, 0] += p_t_i

    p_t /= np.sum(p_t)
    return p_t


def align_graphs(
    graphs: List[np.ndarray], padding: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree
            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
            # normalized_node_degrees.append(normalized_node_degree)
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num
