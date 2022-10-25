"""Exact permutation distance between block graphons."""
import numpy as np

from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.utils.utils_maths import generate_all_permutations
from pygraphon.utils.utils_matrix import permute_matrix


def permutation_distance(
    graphon1: StepGraphon, graphon2: StepGraphon, norm: str = "MSE", exchangeable: bool = True
) -> float:
    """Exact permutation distance between two graphons.

    Implement the mean squared error and mean absolute error for stepgraphons of same size (same number of blocks
    and same areas of blocks).

    Parameters
    ----------
    graphon1 : StepGraphon
        first stepgraphon to compare
    graphon2 : StepGraphon
        second stepgraphon to compare
    norm : str
        ["MAE","MSE"]. Defaults to "MSE".
    exchangeable : bool
        if sets to true, the norm will try all possible permutations of the blocks to
        find the lowest distance. Otherwise assume correspondance between the blocks of the first and second
        graphon. Defaults to True.

    Returns
    -------
    float
         distance between the two graphons

    Raises
    ------
    NotImplementedError
        different number of blocks
    NotImplementedError
       different bandwithHist
    NotImplementedError
        if heteregenous block size
    ValueError
        if norm not in MAE or MSE
    TypeError
        if the graphons are not stepgraphons
    """
    if not isinstance(graphon1, StepGraphon) or not isinstance(graphon2, StepGraphon):
        raise TypeError("graphons should be stepgraphons")

    if norm not in ["MAE", "MSE"]:
        raise ValueError(f"norm should be MAE or MSE, but got {norm}")

    # get the data we need
    graphon1_matrix = graphon1.graphon
    graphon2_matrix = graphon2.graphon
    if graphon1_matrix.shape != graphon2_matrix.shape:
        raise NotImplementedError(
            "Cannot compare two graphons with different number of blocks:"
            + f"{graphon1_matrix.shape} and {graphon2_matrix.shape}"
        )

    # check that graphons have the same bandwidth for their blocks
    # this means they have the same size of blocks if the graphon have the
    # same shape
    if graphon2.bandwidthHist != graphon1.bandwidthHist:
        raise NotImplementedError("different size of graphons cannot be compared for now")

    # check if the graphon blocks all have same size
    if len(np.unique(graphon1.areas)) != 1 or len(np.unique(graphon2.areas)) != 1:
        raise NotImplementedError("Cannot compare graphons with heterogeneous block sizes")

    # generate all possible permutations
    permutations_possible = generate_all_permutations(graphon1_matrix.shape[0])

    norm_value = np.sqrt(np.sum(((graphon1_matrix - graphon2_matrix) ** 2) * graphon1.areas))
    if not exchangeable or norm_value == 0:
        return norm_value

    norm_function = _mse_graphon if norm == "MSE" else _mae_graphon

    values = [norm_value]
    for permutation in permutations_possible:
        graphon2_permuted = permute_matrix(graphon2_matrix, permutation)
        result = norm_function(graphon1_matrix, graphon2_permuted, graphon1.areas)
        values.append(result)
        if result == 0:
            break

    return min(values)


def _mse_graphon(graphon1_matrix, graphon2_matrix, areas):
    """Compute the MSE between two graphons.

    Parameters
    ----------
    graphon1_matrix : np.ndarray
        first graphon
    graphon2_matrix : np.ndarray
        second graphon
    areas : np.ndarray
        areas of the blocks

    Returns
    -------
    float
        MSE between the two graphons
    """
    return np.mean(((graphon1_matrix - graphon2_matrix) ** 2) * areas)


def _mae_graphon(graphon1_matrix, graphon2_matrix, areas):
    """Compute the MAE between two graphons.

    Parameters
    ----------
    graphon1_matrix : np.ndarray
        first graphon
    graphon2_matrix : np.ndarray
        second graphon
    areas : np.ndarray
        areas of the blocks

    Returns
    -------
    float
        MAE between the two graphons
    """
    return np.mean(np.abs(graphon1_matrix - graphon2_matrix) * areas)
