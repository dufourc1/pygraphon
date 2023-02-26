"""Implementation of network histogram estimator."""
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from pygraphon.utils.utils_graph import edge_density

from .assignment import Assignment

EPS = np.spacing(1)


def greedy_opt(
    A: np.ndarray,
    inputLabelVec: np.ndarray,
    absTol: float = 2.5 * 1e-4,
    maxNumIterations: int = 500,
    past_non_improving: int = 3,
    *args,
    **kwargs,
) -> Assignment:
    """Greedy optimization of the network histogram.

    This function reproduces the greedy optimization of the network histogram paper,
    the two for loops are not necessary but are kept for sake of comparison with the original code.

    Parameters
    ----------
    A : np.ndarray
        adjacency matrix
    inputLabelVec : np.ndarray
        initial guess of the node membership (block labels)
    absTol : float
        absolute tolerance for convergence, by default 2.5 * 1e-4
    maxNumIterations : int
        number of steps, by default 500

    Returns
    -------
    Assignment
        optimized node membership
    """
    n = A.shape[0]
    n_obs = int(n * (n - 1) / 2)

    # Initialize the assignment
    best_assignment = Assignment(inputLabelVec, A)
    current_assignment = deepcopy(best_assignment)
    overall_best = deepcopy(best_assignment)

    rho = edge_density(A)
    step_internal = 2 * 10**4 if n > 256 else n_obs

    past_best_likelihood = [overall_best.log_likelihood]

    pbar = tqdm(range(maxNumIterations))
    for m in pbar:
        index_i = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)
        index_j = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)
        index_k = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)

        # decide if we do one or two swaps before checking if we improved
        one_or_two_swaps = np.array(np.random.uniform(size=step_internal) > 2 / 3) + 1

        pbar.set_description(f"Log likelihood: {best_assignment.log_likelihood/n_obs:.4f}")

        for s in range(step_internal):
            updated = False
            for swap_number in range(one_or_two_swaps[s]):
                if swap_number == 0:
                    if (index_i[s] != index_j[s]) and (
                        current_assignment.labels[index_i[s]]
                        != current_assignment.labels[index_j[s]]
                    ):
                        current_assignment.update((index_i[s], index_j[s]), A)
                        updated = True
                if swap_number == 1:
                    if (index_j[s] != index_k[s]) and (
                        current_assignment.labels[index_j[s]]
                        != current_assignment.labels[index_k[s]]
                    ):
                        current_assignment.update((index_j[s], index_k[s]), A)
                        updated = True

            if updated:
                if current_assignment.log_likelihood > best_assignment.log_likelihood:
                    best_assignment.copy_from_other(current_assignment)
                else:
                    current_assignment.copy_from_other(best_assignment)

        if best_assignment.log_likelihood > overall_best.log_likelihood:
            overall_best.copy_from_other(best_assignment)
        else:
            best_assignment.copy_from_other(overall_best)
            current_assignment.copy_from_other(overall_best)

        past_best_likelihood.append(overall_best.log_likelihood)

        if m % 5 == 0 and m > past_non_improving:
            if np.all(
                (
                    np.array(past_best_likelihood[-1 - past_non_improving : -1])
                    - np.array(past_best_likelihood[-past_non_improving:])
                )
                / rho
                < absTol
            ):
                break

    return overall_best
