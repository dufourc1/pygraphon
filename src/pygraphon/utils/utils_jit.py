import numba as nb
import numpy as np


@nb.jit(nopython=True)
def log_likelihood(probs: np.ndarray, A: np.ndarray, eps=1e-8) -> float:
    r"""Compute the log-likelihood of the graphon given the adjacency matrix of a simple graph.

    .. math::
        \sum_{i<j} \log(\theta_{ij}){A_{ij}} + \log(1-\theta_{ij}){1-A_{ij}}


    .. math::
        \theta_{ij} =  \begin{cases} \epsilon & \text{if } \theta_{ij} < \epsilon \\
        p_{ij} & \text{if } \epsilon \leq \theta_{ij} \leq 1 - \epsilon \\
        1 - \epsilon & \text{if } \theta_{ij} > 1 - \epsilon \end{cases},

    where :math:`A_{ij}` is :py:obj:`A[i,j]`, :math:`p_{ij}` is :py:obj:`probs[i,j]`,
    and :math:`\epsilon` is :py:obj:`np.spacing(1)`.

    Parameters
    ----------
    probs : np.ndarray
        edge probability matrix
    A : np.ndarray
        adjacency matrix of the graph

    Raises
    ------
    ValueError
        if the probability matrix contains values greater than 1

    Returns
    -------
    float
        log-likelihood of the graphon given the adjacency matrix

    .. note::
        Suppose :py:obj:`probs` and :py:obj:`A` are aligned, i.e. :py:obj:`probs[i,j]` is the probability
        of an edge between node :math:`i` and node :math:`j` and :py:obj:`A[i,j]` is the adjacency
        of the edge between node :math:`i` and node :math:`j`.

        This function assumes the graph is simple, i.e. only undirected edges and no self-loops.
        For this reason, only the upper triangular part of :py:obj:`probs` and :py:obj:`A` are considered.
    """
    result = 0
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            x = A[i, j]
            p = probs[i, j]
            if p > 1:
                raise ValueError("Probability matrix contains values greater than 1")
            if p < eps:
                result += x * np.log(eps)
                result += (1 - x) * np.log(1 - eps)
            elif p > 1 - eps:
                result += x * np.log(1 - eps)
                result += (1 - x) * np.log(eps)
            else:
                result += x * np.log(p)
                result += (1 - x) * np.log(1 - p)
    return result
