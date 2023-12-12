"""Fast assignment class."""
from math import log
from typing import Tuple

import numba as nb
import numpy as np

from pygraphon.utils import EPS
from pygraphon.utils.utils_graph import compute_edge_between_groups


class Assignment:
    def __init__(self, labels: np.ndarray, adjacency: np.ndarray) -> None:
        self.labels = labels - np.min(labels)  # 0-indexed
        self.labels = self.labels.astype(int)
        self.sample_size = adjacency.shape[0] * (adjacency.shape[0] - 1) / 2
        self.num_groups = len(np.unique(self.labels.astype(int)))
        self.group_sizes: np.ndarray = np.bincount(self.labels.astype(int))
        self.counts, self.realized = self.compute_counts_realized(adjacency)
        self.theta = self.realized / self.counts
        self.log_likelihood = self.compute_log_likelihood()

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
            member_g = np.where(self.labels == g)[0]
            for g_prime in range(g + 1, self.num_groups):
                counts[g, g_prime] = self.group_sizes[g] * self.group_sizes[g_prime]
                member_g_prime = np.where(self.labels == g_prime)[0]
                realized[g, g_prime] = compute_edge_between_groups(
                    adjacency, member_g, member_g_prime
                )
                counts[g_prime, g] = counts[g, g_prime]
                realized[g_prime, g] = realized[g, g_prime]
            counts[g, g] = self.group_sizes[g] * (self.group_sizes[g] - 1) / 2
            realized[g, g] = compute_edge_between_groups(adjacency, member_g, member_g) / 2

        return counts, realized

    def compute_log_likelihood(self, scale: bool = False) -> float:
        """Compute the likelihood of the assignment.

        Parameters
        ----------
        scale : bool
            Whether to scale the likelihood by the sample size, by default False

        Returns
        -------
        float
            log likelihood
        """
        scaling = 1 / np.sum(np.triu(self.realized)) if scale else 1
        return np.sum(bernlikelihood(self.theta) * np.triu(self.counts)) * scaling  # type: ignore

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
        # important: we need to pass the unswapped labels to compute_counts_realized
        self.realized = update_realized_number(self.realized, self.labels, adjacency, swap)
        self.labels[swap[0]], self.labels[swap[1]] = (
            self.labels[swap[1]],
            self.labels[swap[0]],
        )
        self.theta = self.realized / self.counts

        self.log_likelihood = self.compute_log_likelihood()

    def copy_from_other(self, other):
        self.theta = np.copy(other.theta)
        self.log_likelihood = other.log_likelihood
        self.labels = np.copy(other.labels)
        self.realized = np.copy(other.realized)

    def labels_to_latent_variables(self) -> np.ndarray:
        """Convert the labels to latent variables for a stochastic block model.

        The nodes in the same community are assigned to the same latent variable.

        Returns
        -------
        np.ndarray
            Latent variables
        """
        n = self.labels.shape[0]
        latent_variables = np.zeros_like(self.labels, dtype=float)
        offset = 0

        for i in range(self.num_groups):
            group_size_0_1 = self.group_sizes[i] / n
            latent_variables[np.where(self.labels == i)] = offset + group_size_0_1 / 2
            offset += group_size_0_1
        return latent_variables


#########################
# Numba compiled functions
#########################


@nb.jit(nopython=True)
def update_realized_number(realized_edges, labels, adjacency, swap):
    """Update the realized_edges  between communities after a swap of two node labels.

    The two node labels are supposed to be different.

    Parameters
    ----------
    realized_edges : np.ndarray
        realized_edges between communities
    labels : np.ndarray
        Labels of the nodes before the swap
    adjacency : np.ndarray
        Adjacency matrix of the graph
    swap : Tuple[int, int]
        The two node labels that were swapped

    Returns
    -------
    np.ndarray
        Updated realized_edges
    """
    group_node_1 = labels[swap[0]]
    group_node_2 = labels[swap[1]]

    for i in range(adjacency.shape[0]):
        if i == swap[0] or i == swap[1]:
            continue

        if adjacency[swap[0], i] == adjacency[swap[1], i]:
            continue

        group_i = labels[i]
        if adjacency[swap[0], i] == 1:
            realized_edges[group_node_1, group_i] -= 1
            realized_edges[group_i, group_node_1] = realized_edges[group_node_1, group_i]

            realized_edges[group_node_2, group_i] += 1
            realized_edges[group_i, group_node_2] = realized_edges[group_node_2, group_i]
        if adjacency[swap[1], i] == 1:
            realized_edges[group_node_2, group_i] -= 1
            realized_edges[group_i, group_node_2] = realized_edges[group_node_2, group_i]

            realized_edges[group_node_1, group_i] += 1
            realized_edges[group_i, group_node_1] = realized_edges[group_node_1, group_i]

    return realized_edges


@nb.vectorize("float64(float64)", nopython=True)
def bernlikelihood(x):
    """Compute x * log(x).

    Parameters
    ----------
    x : np.ndarray
        Array of values

    Returns
    -------
    np.ndarray
        x * log(x)
    """
    if x <= EPS:
        return (EPS) * log(EPS) + (1 - EPS) * log(1 - EPS)
    elif x >= 1 - EPS:
        return (1 - EPS) * log(1 - EPS) + EPS * log(EPS)
    else:
        return x * log(x) + (1 - x) * log(1 - x)
