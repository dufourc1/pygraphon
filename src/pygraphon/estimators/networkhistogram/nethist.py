"""Implementation of network histogram estimator."""
import random
from copy import deepcopy
from typing import List, Optional, Tuple

import numba as nb
import numpy as np
import scipy
import scipy.sparse.linalg
from scipy import stats
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from pygraphon.utils.utils_graph import check_simple_adjacency_matrix

from .graphest_fastgreedy import graphest_fastgreedy


class Assignment:
    def __init__(self, labels: np.ndarray, adjacency: np.ndarray) -> None:
        self.labels = labels
        self.num_groups = len(np.unique(self.labels.astype(int)))
        self.group_sizes: np.ndarray = np.bincount(self.labels.astype(int))
        self.counts, self.realized = self.compute_counts_realized(adjacency)
        self.theta = self.realized / self.counts
        self.likelihood = self.compute_log_likelihood()

    def compute_counts_realized(self, adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the number of edges and the number of realized edges between communities specified by labels.

        Parameters
        ----------
        adjacency : np.ndarray
            Adjacency matrix of the graph

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            counts, realized
        """
        counts = np.zeros((self.num_groups, self.num_groups))
        realized = np.zeros_like(counts)
        for g in range(self.num_groups):
            for g_prime in range(g + 1, self.num_groups):
                counts[g, g_prime] = self.group_sizes[g] * self.group_sizes[g_prime]
                realized[g, g_prime] = np.sum(
                    adjacency[self.labels == g, :][:, self.labels == g_prime]
                )
                realized[g_prime, g] = realized[g, g_prime]
                counts[g_prime, g] = counts[g, g_prime]
            counts[g, g] = self.group_sizes[g] * (self.group_sizes[g] - 1) / 2
            realized[g, g] = np.sum(adjacency[self.labels == g, :][:, self.labels == g]) / 2

        return counts, realized

    def compute_log_likelihood(self) -> float:
        """Compute the likelihood of the assignment.

        Returns
        -------
        float
            log likelihood
        """
        return _log_likelihood(self.num_groups, self.theta, self.counts)

    def update(self, swap: Tuple[int, int], adjacency: np.ndarray) -> None:
        """Update the assignment after a swap of two node labels.

        Parameters
        ----------
        swap : Tuple[int, int]
            The two node labels that were swapped
        adjacency : np.ndarray
            Adjacency matrix of the graph

        Returns
        -------
        None
        """
        self.labels[swap[0]], self.labels[swap[1]] = (
            self.labels[swap[1]],
            self.labels[swap[0]],
        )
        for g in range(self.num_groups):
            for g_prime in [int(self.labels[swap[0]]), int(self.labels[swap[1]])]:
                self.realized[g, g_prime] = np.sum(
                    adjacency[self.labels == g, :][:, self.labels == g_prime]
                )
                if g == g_prime:
                    self.realized[g, g_prime] /= 2
                self.realized[g_prime, g] = self.realized[g, g_prime]

        self.theta = self.realized / self.counts
        self.likelihood = self.compute_log_likelihood()


@nb.jit(nopython=True)
def _log_likelihood(num_groups: int, theta, count) -> float:
    """Compute the log likelihood of the assignment.

    Parameters
    ----------
    num_groups : int
        Number of groups
    theta : np.ndarray
        Estimated probabilities
    count : np.ndarray
        Number of edges between groups

    Returns
    -------
    float
        log likelihood
    """
    loglike = 0
    for g in range(num_groups):
        for g_prime in range(g, num_groups):
            theta_hat = (
                1e-16
                if theta[g, g_prime] < 1e-16
                else (1 - 1e-16 if theta[g, g_prime] > 1 - 1e-16 else theta[g, g_prime])
            )
            loglike += (
                theta_hat * np.log(theta_hat) + (1 - theta_hat) * np.log(1 - theta_hat)
            ) * count[g, g_prime]
    return loglike


def _oracle_analysis_badnwidth(A: np.ndarray, type_: str = "degs", alpha: float = 1) -> float:
    c = min(4, np.sqrt(A.shape[0]) / 8)
    h, _ = oracbwplugin(A=A, c=c, type_=type_, alpha=alpha)
    return int(h)


def oracbwplugin(
    A: np.ndarray, c: float, type_: str = "degs", alpha: float = 1
) -> Tuple[float, float]:
    """Oracle bandwidth plug-in estimtor for network histograms.

    The call h = oracbwplugin(A,c,type_,alpha) returns a plug-in estimate
    of the optimal histogram bandwidth (blockmodel community size) as a
    function of the following inputs

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (must be a simple graph)
    c : float
        positive multiplier by which to estimate slope 1/- sqrt(n)
    type_ : str
         Estimate slope from sorted vector ('degs' or 'eigs'). Defaults to "degs".
    alpha : float
         Holder exponent. Defaults to 1.

    Returns
    -------
    Tuple[float, float]
        optimal histogram bandwidth, estimated mean squared error

    Raises
    ------
    ValueError
        if c is not positive
    NotImplementedError
        alpha not 1
    NotImplementedError
        type_ not degs or eigs

    Examples
    --------
    >> h, _ = oracbwplugin(A,3,'eigs',1); # returns h = 73.5910

    >> h, _ = oracbwplugin(A,3,'degs',1); # returns h = 74.1031
    """
    # input checks
    if c <= 0:
        raise ValueError("c must be positive")
    if alpha != 1:
        raise NotImplementedError("Currently only support alpha = 1")
    check_simple_adjacency_matrix(A)

    # conversion for scipy functions
    A = A.astype(float)

    n = A.shape[0]
    midPt = np.arange(round(n / 2 - c * np.sqrt(n)) - 1, round(n / 2 + c * np.sqrt(n)))
    rhoHat = np.sum(A) / (n * (n - 1))
    pseudo_inverse_rho_hat = 1 / rhoHat if rhoHat != 0 else 0

    if type_ == "eigs":
        mult, u = scipy.sparse.linalg.eigs(A, 1, which="LR")
        u = u.real.ravel()
    elif type_ == "degs":
        u = np.sum(A, axis=1)
        mult = (u.T @ A @ u) / np.sum(u * u) ** 2
    else:
        raise NotImplementedError(f"Unknown input type_: {type_}")

    # if all the degrees are the same, there is no information in there
    if np.unique(u).size == 1:
        h = 1
        estMSqrd = 0
    # if the slope is 0, we have an issue: we need to have a non uniform array
    # of degrees to compute the slope: we augment the value of c to get a non uniform array
    # if needed
    else:
        u = np.sort(u.ravel())
        uMid = u[midPt]
        increment = 1
        while np.unique(uMid).size == 1:
            midPt = np.arange(
                round(n / 2 - c * np.sqrt(n)) - 1 - increment,
                round(n / 2 + c * np.sqrt(n)) + increment,
            )
            uMid = u[midPt]
            increment += 1
        reg = stats.linregress(np.arange(1, len(uMid) + 1), uMid)
        p = (reg[1], reg[0])
        h = (
            2 ** (alpha + 1)
            * alpha
            * mult**2
            * (p[0] + p[1] * len(uMid) / 2) ** 2
            * p[1] ** 2
            * pseudo_inverse_rho_hat
        ) ** (-1 / (2 * (alpha + 1)))
        estMSqrd = (
            2
            * mult**2
            * (p[0] + p[1] * len(uMid) / 2) ** 2
            * p[1] ** 2
            * pseudo_inverse_rho_hat**2
            * (n + 1) ** 2
        )
    # MISEfhatBnd = estMSqrd * ((2 / np.sqrt(estMSqrd)) * (sampleSize * rhoHat) ** (-1 / 2) + 1 / n)
    return h, estMSqrd


def _first_guess_blocks(A: np.ndarray, h: int, regParam: float) -> np.ndarray:
    """Compute the first guess of the node membership (block labels).

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    h : int
        number of nodes per block
    regParam : float
        regularisation parameter to help decomposing the adjacency matrix

    Returns
    -------
    np.ndarray
        first guess of the node membership (block labels)
    """
    n = A.shape[0]

    if regParam == 0:
        distVec = pairwise_distances(A, A, metric="manhattan") / n
    else:
        A_inter = A + regParam * np.ones_like(A) * regParam
        distVec = pairwise_distances(A_inter, A_inter, metric="manhattan") / n
    L = np.ones_like(distVec) - distVec**2
    d = np.sum(L, axis=1)
    d = d[:, np.newaxis]
    L_inter = np.outer(d**-0.5, d**-0.5) * L - np.outer(np.sqrt(d), np.sqrt(d)) / np.sqrt(
        np.sum(d**2)
    )
    _, u = scipy.sparse.linalg.eigs(L_inter, k=1, which="LR")
    u = u.real.ravel()

    # set 1st coord >= 0 wlog, to fix an arbitrary sign permutation
    u = u * np.sign(u[0])
    # sort on this embedding in ascending fashion
    ind = u.argsort().astype(int)
    k = int(np.ceil(n / h))

    # Assign initial label vector from row-similarity ordering
    # this is correct
    idxInit = np.zeros(n)
    for i in range(k):
        idxInit[ind[i * h : min(n, (i + 1) * h)]] = i
    return idxInit


def nethist(
    A: np.ndarray, h: int = None, verbose: bool = False, trace: bool = False
) -> Tuple[List, int, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Compute the network histogram of an N-by-N adjacency matrix.

    adjacency matrix is assumed to be  0-1 valued, symmetric, and with zero on the diagonal.

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    h : int
        specifies the number of nodes in each histogram bin,, by default is optimized based on input.
    verbose : bool
        if True logs progress of optimization , by default False
    trace : bool
        if True trace the optimization, by default False

    Returns
    -------
    Tuple[List, int, Optional[Tuple[np.ndarray, np.ndarray]]]
        [idx,h] return the vector of group membership of the nodes and the parameter h
    """
    check_simple_adjacency_matrix(A)

    # conversion for scipy functions
    A = A.astype(float)

    # Compute necessary summaries from A
    n = A.shape[0]
    rhoHat = np.sum(A) / (n * (n - 1))

    # use data driven h
    h = int(h) if h is not None else int(_oracle_analysis_badnwidth(A=A))

    # min h i s 2 so that no node is in its own group
    h = max(2, min(n, np.round(h)))

    lastGroupSize = n % h

    # step down h to avoid singleton group
    while lastGroupSize == 1 and h > 2:
        h = h - 1
        lastGroupSize = n % h

    idxInit = _first_guess_blocks(A, h, regParam=rhoHat / 4)
    return greedy_opt(A, idxInit), h, None
    if trace:
        idx, _, trace = graphest_fastgreedy(
            A=A, hbar=h, inputLabelVec=idxInit, verbose=verbose, trace=trace
        )
    else:
        idx, _ = graphest_fastgreedy(
            A=A, hbar=h, inputLabelVec=idxInit, verbose=verbose, trace=trace
        )
    return idx, h, trace


def greedy_opt(
    A: np.ndarray,
    inputLabelVec: np.ndarray,
    absTol: float = 2.5 * 1e-4,
    maxNumRestarts: int = 500,
    *args,
    **kwargs,
) -> np.ndarray:
    """Greedy optimization of the network histogram.

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    inputLabelVec : np.ndarray
        initial guess of the node membership (block labels)
    absTol : float
        absolute tolerance for convergence, by default 2.5 * 1e-4
    maxNumRestarts : int
        number of steps, by default 500

    Returns
    -------
    np.ndarray
        optimized node membership (block labels)
    """
    n = A.shape[0]
    sample_size = int(n * (n - 1) / 2)
    normalized_tol = absTol / (sample_size / np.sum(A))
    # Initialize the assignment
    best_assignment = Assignment(inputLabelVec, A)
    current_assignment = deepcopy(best_assignment)

    for _ in tqdm(range(maxNumRestarts)):
        i, j = random.sample(range(n), 2)
        current_assignment.update((i, j), A)
        if current_assignment.likelihood > best_assignment.likelihood:
            if current_assignment.likelihood - best_assignment.likelihood <= normalized_tol:
                best_assignment = deepcopy(current_assignment)
                break
            else:
                best_assignment = deepcopy(current_assignment)
        else:
            current_assignment = deepcopy(best_assignment)

    return best_assignment.labels
