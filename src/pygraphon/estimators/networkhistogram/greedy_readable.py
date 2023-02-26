"""Implementation of network histogram estimator."""
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from .assignment import Assignment

EPS = np.spacing(1)


def greedy_opt(
    A: np.ndarray,
    inputLabelVec: np.ndarray,
    absTol: float = 2.5 * 1e-4,
    maxNumRestarts: int = 500,
    *args,
    **kwargs,
) -> Assignment:
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
    Assignment
        optimized node membership
    """
    n = A.shape[0]
    n_obs = int(n * (n - 1) / 2)

    # Initialize the assignment
    best_assignment = Assignment(inputLabelVec, A)
    current_assignment = deepcopy(best_assignment)
    overall_best = deepcopy(best_assignment)

    step_internal = 2 * 10**4 if n > 256 else n_obs

    tol_small_improvement = 0
    nb_improvements = 0
    global_non_improvements = 0

    pbar = tqdm(range(maxNumRestarts))
    for m in pbar:
        index_i = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)
        index_j = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)
        index_k = np.ceil(np.random.uniform(low=-1, high=n - 1, size=step_internal)).astype(int)
        one_or_two_swaps = np.array(np.random.uniform(size=step_internal) > 2 / 3) + 1

        pbar.set_description(f"Likelihood: {best_assignment.log_likelihood/n_obs:.4f}")
        before_inner_step = best_assignment.log_likelihood
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
                    nb_improvements += 1
                    if current_assignment.log_likelihood - best_assignment.log_likelihood < absTol:
                        tol_small_improvement += 1
                    else:
                        tol_small_improvement = 0
                    best_assignment.copy_from_other(current_assignment)
                else:
                    current_assignment.copy_from_other(best_assignment)

                if tol_small_improvement >= 3:
                    break

        if best_assignment.log_likelihood > overall_best.log_likelihood:
            overall_best.copy_from_other(best_assignment)
        else:
            best_assignment.copy_from_other(overall_best)
            current_assignment.copy_from_other(overall_best)

        if before_inner_step >= best_assignment.log_likelihood:
            global_non_improvements += 1
        else:
            global_non_improvements = 0
        if global_non_improvements >= 3:
            break
    return overall_best
