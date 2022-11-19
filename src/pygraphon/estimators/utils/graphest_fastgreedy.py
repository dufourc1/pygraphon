"""Greedy optimization procedure for the nethist estimator."""
from typing import Iterable, Optional, Tuple

import numpy as np
import numpy.random as rnd
import scipy
from numba import njit
from tqdm import tqdm

from pygraphon.utils.utils_matrix import (
    bound_away_from_one_and_zero_arrays,
    upper_triangle_values,
)


def graphest_fastgreedy(
    A: np.ndarray,
    hbar: int,
    inputLabelVec: Iterable,
    absTol: float = 2.5 * 1e-4,
    maxNumRestarts: int = 500,
    verbose: bool = True,
    trace: bool = True,
) -> Tuple[np.ndarray, int, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Implement likelihood-based optimization for nethist.

    Returns a list of cluster labels.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (simple graph)
    hbar : int
        number of nodes in each cluster
    inputLabelVec : Iterable
        initial guess for the cluster labels.
    absTol : float
        when to stop optimizing ll, by default 2.5*1e-4
    maxNumRestarts : int
        maximum number of restart for the greedy algorithm, by default 500
    verbose : bool
        verbose flag, by default True
    trace : bool
        trace flag, by default True

    Returns
    -------
    Tuple[np.ndarray, int, Optional[Tuple[np.ndarray, np.ndarray]]]
        cluster labels, number of blocks,
        and optionally the trace of the optimization (lls_trace, normalized_best_ll_trace)

    Raises
    ------
    RuntimeError
       if a cluster has only one node
    RuntimeError
        if a node is not assigned to any cluster
    """
    n = A.shape[0]
    numGreedySteps, allInds = set_num_greedy_steps(n)

    sampleSize = int(n * (n - 1) / 2)
    smaller_last_group = 0 if n % hbar == 0 else 1
    k = int(np.ceil(n / hbar))
    equalSizeInds = np.arange(0, (k - smaller_last_group) - 1)
    orderedLabels = np.zeros((n, 1))
    h = np.zeros(k)
    hbar = int(hbar)
    orderedClusterInds = np.zeros((k, hbar))
    for a in range(k - smaller_last_group):
        orderedInds = np.arange(a * hbar, (a + 1) * hbar)
        h[a] = len(orderedInds)
        orderedLabels[orderedInds - 1] = a
        orderedClusterInds[a, :] = orderedInds
    if smaller_last_group:
        orderedIndsLast = np.arange((k - 1) * hbar, n)
        h[k - 1] = len(orderedIndsLast)
        orderedClusterInds[k - 1, :] = np.concatenate(
            [orderedIndsLast, np.zeros(hbar - len(orderedIndsLast))]
        )

    # number of possible connections between the clusters
    habSqrd = np.outer(h, h) - np.diag(np.multiply(h, h) - np.multiply(h, (h - 1)) / 2)

    if np.all(habSqrd) < 1:
        raise RuntimeError("All clusters must contain at least 2 nodes")
    if np.max(orderedClusterInds) != n - 1 or np.sum(h) != n:
        raise RuntimeError("All nodes must be assigned to a cluster")

    initialLabelVec = inputLabelVec.astype(int)
    initialClusterInds = np.zeros((k, hbar), dtype=int)
    initialClusterCentroids = np.zeros((k, n))

    for a in range(k - smaller_last_group):
        initialClusterInds[a, :] = np.where(initialLabelVec == a)[0]
        initialClusterCentroids[a, :] = np.sum(A[:, initialClusterInds[a, :]], axis=1)

    if smaller_last_group:
        initialClusterInds[k - 1, 0 : len(np.where(initialLabelVec == k - 1)[0])] = np.where(
            initialLabelVec == k - 1
        )[0]
        initialClusterInds[k - 1, len(np.where(initialLabelVec == k - 1)[0]):] = -2

    # matrix of size (K,K) with the number of edges between clusters
    initialACounts = _getSampleCounts(A, initialClusterInds)

    initialLL = _fastNormalizedBMLogLik(
        upper_triangle_values(initialACounts) / upper_triangle_values(habSqrd),
        upper_triangle_values(habSqrd),
        sampleSize,
    )

    bestLL = initialLL
    oldNormalizedBestLL = bestLL * 2 * sampleSize / np.sum(A)

    # if verbose:
    #    print(f"Initial log-likelihood: {bestLL:.4f}")
    #    print(f"Initial normalized log-likelihood: {oldNormalizedBestLL:.4f}")

    bestLabelVec = initialLabelVec
    bestCount = 0
    conseczeroImprovement = 0
    tolCounter = 0

    if trace:
        lls_trace = [bestLL]
        normalized_best_ll_trace = [oldNormalizedBestLL]

    if verbose:
        pbar = tqdm(range(maxNumRestarts))
    else:
        pbar = range(maxNumRestarts)

    for mm in pbar:
        oneTwoVec = np.array(rnd.uniform(size=numGreedySteps) > 2 / 3) + np.ones(
            numGreedySteps, dtype=int
        )

        # random integers between 0 and n-1
        iVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)
        kVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)
        jVec = np.ceil(rnd.uniform(size=numGreedySteps) * n - 1).astype(int)

        bestClusterInds = np.zeros(shape=(k, hbar), dtype=int)
        for a in range(k - smaller_last_group):
            bestClusterInds[a, :] = np.where(bestLabelVec == a)[0]

        if smaller_last_group:
            bestClusterInds[k - 1, 0 : len(np.where(bestLabelVec == k - 1)[0])] = np.where(
                bestLabelVec == k - 1
            )[0]

        bestACounts = _getSampleCounts(A, bestClusterInds)
        bestLL = _fastNormalizedBMLogLik(
            upper_triangle_values(bestACounts) / upper_triangle_values(habSqrd),
            upper_triangle_values(habSqrd).ravel(),
            sampleSize,
        )

        currentACounts = bestACounts
        currentClusterInds = bestClusterInds
        currentLL = bestLL
        currentLabelVec = np.copy(bestLabelVec)

        for m in range(numGreedySteps):

            # prepare to update quantities for trial clustering
            trialClusterInds = np.copy(currentClusterInds)
            trialLabelVec = np.copy(currentLabelVec)
            trialACounts = np.copy(currentACounts)
            trialLL = currentLL

            # implement consecutive pairwise swaps to obtain trial clustering
            for swapNum in range(oneTwoVec[m]):
                if swapNum == 0:
                    i = iVec[m]
                    j = jVec[m]
                    a = trialLabelVec[i]
                    b = trialLabelVec[j]

                elif a != b:
                    i = jVec[m]
                    j = kVec[m]
                    a = trialLabelVec[i]
                    b = trialLabelVec[j]

                # swap and update the trial likelihood only if nodes i and j are in different clusters
                if a != b:

                    trialLabelVec[i], trialLabelVec[j] = b, a

                    habSqrdCola = habSqrd[:, a]
                    habSqrdColb = habSqrd[:, b]
                    habSqrdEntryab = np.array(habSqrd[a, b])

                    oldThetaCola = trialACounts[:, a] / habSqrdCola
                    oldThetaColb = trialACounts[:, b] / habSqrdColb
                    oldThetaEntryab = np.array(trialACounts[a, b] / habSqrdEntryab)

                    (
                        oldThetaCola,
                        oldThetaColb,
                        oldThetaEntryab,
                    ) = bound_away_from_one_and_zero_arrays(
                        [oldThetaCola, oldThetaColb, oldThetaEntryab]
                    )

                    # begin updating
                    # replace cluster inds i  in row a  by j and vice versa for b
                    trialClusterInds[a, trialClusterInds[a, :] == i] = j
                    trialClusterInds[b, trialClusterInds[b, :] == j] = i

                    ARowiMinusRowj = A[i, :] - A[j, :]

                    # concatenate all possible group indices into a a matrix
                    clusterIndMat = trialClusterInds[equalSizeInds, :]
                    sumAijc = np.sum(ARowiMinusRowj[clusterIndMat], axis=1)
                    trialACounts[equalSizeInds, a] = trialACounts[equalSizeInds, a] - sumAijc
                    trialACounts[equalSizeInds, b] = trialACounts[equalSizeInds, b] + sumAijc

                    if smaller_last_group:  # take care of last group separately if unequal size
                        sumAijEnd = np.sum(
                            ARowiMinusRowj[trialClusterInds[k - 1, trialClusterInds[k - 1, :] > 0]]
                        )
                        trialACounts[k - 1, a] = trialACounts[k - 1, a] - sumAijEnd
                        trialACounts[k - 1, b] = trialACounts[k - 1, b] + sumAijEnd

                    # update the above for special case c==a | c==b
                    trialACounts[a, a] = trialACounts[a, a] + A[i, j]
                    trialACounts[b, b] = trialACounts[b, b] + A[i, j]
                    if smaller_last_group and b == k:
                        trialACounts[a, b] = (
                            trialACounts[a, b]
                            - np.sum(
                                ARowiMinusRowj[trialClusterInds[b, trialClusterInds[b, :] > 0]]
                            )
                            - 2 * A[i, j]
                        )
                    else:
                        trialACounts[a, b] = (
                            trialACounts[a, b]
                            - np.sum(ARowiMinusRowj[trialClusterInds[b, :]])
                            - A[i, j]
                        )
                    trialACounts[b, a] = trialACounts[a, b]

                    # Normalize and respect symmetry of trialAbar matrix
                    trialACounts[a, :] = trialACounts[:, a]
                    trialACounts[b, :] = trialACounts[:, b]

                    # Now calculate changed likelihood directly
                    thetaCola = trialACounts[:, a] / habSqrdCola
                    thetaColb = trialACounts[:, b] / habSqrdColb
                    thetaEntryab = np.array(trialACounts[a, b] / habSqrdEntryab)

                    thetaCola, thetaColb, thetaEntryab = bound_away_from_one_and_zero_arrays(
                        [thetaCola, thetaColb, thetaEntryab]
                    )

                    # For this to work, we will have had to subtract out terms prior to updating
                    deltaNegEnt = _delta_neg(
                        habSqrdCola, thetaCola, habSqrdColb, thetaColb, habSqrdEntryab, thetaEntryab
                    )
                    oldDeltaNegEnt = _delta_neg(
                        habSqrdCola,
                        oldThetaCola,
                        habSqrdColb,
                        oldThetaColb,
                        habSqrdEntryab,
                        oldThetaEntryab,
                    )

                    # Update the log-likelihood - O(k)
                    trialLL = trialLL + (deltaNegEnt - oldDeltaNegEnt) / sampleSize

            # Metropolis or greedy step; if trial clustering accepted, then update current <- trial
            if trialLL > currentLL:
                currentLabelVec = trialLabelVec
                currentLL = trialLL
                currentACounts = trialACounts
                currentClusterInds = trialClusterInds

        if currentLL > bestLL:
            bestLL = np.copy(currentLL)
            bestLabelVec = np.copy(currentLabelVec)
            bestCount += 1

        if trace:
            lls_trace.append(bestLL)
            normalized_best_ll_trace.append(bestLL * 2 * sampleSize / np.sum(A))

        if mm % 5 == 0:
            normalizedBestLL = bestLL * 2 * sampleSize / np.sum(A)
            if verbose:
                pbar.set_description(
                    f"LL: {normalizedBestLL:.4f},  {bestCount} global improvements"
                )

            if bestCount == 0:
                conseczeroImprovement += 1
            else:
                bestCount = 0
                conseczeroImprovement = 0
            if normalizedBestLL - oldNormalizedBestLL < absTol:
                tolCounter += 1
            else:
                tolCounter = 0
            oldNormalizedBestLL = normalizedBestLL
            # if 3 consecutive likelihood improvements less than specified tolerance break
            if tolCounter >= 3:
                # if verbose:
                #    print(
                #        "3 consecutive likelihood improvements less than specified tolerance; quitting now"
                #   )
                break
            if allInds:
                # local optimum likely reached in random-ordered greedy like likelihood search
                if conseczeroImprovement == 2:
                    # if verbose:
                    #    print(
                    #        "Local optimum likely reached in random-ordered greedy likelihood search; quitting now"
                    #    )
                    break
            else:
                # Local optimum likely reached in random ordere greedy like likelihood search
                if conseczeroImprovement == np.ceil(k * scipy.special.binom(n, 2) / numGreedySteps):
                    # if verbose:
                    #    print(
                    #        "Local optimum likely reached in random-ordered greedy likelihood search; quitting now"
                    #    )
                    break
    if verbose:
        pbar.close()
    if trace:
        return bestLabelVec, k, (lls_trace, normalized_best_ll_trace)
    return bestLabelVec, k


def set_num_greedy_steps(n) -> Tuple[int, bool]:
    """Arbitrary cutoff to the number of greeddy steps to take at each iterations.

    Parameters
    ----------
    n : int
         number of nodes in graph

    Returns
    -------
    Tuple[int, bool]
        number of greedy steps to take, whether to use all indices
    """
    if n <= 256:
        numGreedySteps = int(n * (n - 1) / 2)
        allInds = True
    else:
        # Only a random subset of pairs will be visited on each iteration
        numGreedySteps = 2 * 10**4
        allInds = False
    return numGreedySteps, allInds


# works
@njit
def _delta_neg(habSqrdCola, thetaCola, habSqrdColb, thetaColb, habSqrdEntryab, thetaEntryab):
    return np.sum(
        habSqrdCola * (thetaCola * np.log(thetaCola) + (1 - thetaCola) * np.log(1 - thetaCola))
        + habSqrdColb * (thetaColb * np.log(thetaColb) + (1 - thetaColb) * np.log(1 - thetaColb))
    ) - habSqrdEntryab * (
        thetaEntryab * np.log(thetaEntryab) + (1 - thetaEntryab) * np.log(1 - thetaEntryab)
    )


# works
def _getSampleCounts(X, clusterInds):
    """_summary_

    Args:
        X (_type_): adjacency matrix
        clusterInds (_type_): cluster assignments

    Returns:
        _type_: _description_
    """

    numClusters = clusterInds.shape[0]
    Xsums = np.zeros((numClusters, numClusters), dtype=int)
    for b in range(1, numClusters):
        for a in range(b):  # sum over strict upper-triangular elements
            clusterIndsa = clusterInds[a, clusterInds[a, :] >= 0]
            clusterIndsb = clusterInds[b, clusterInds[b, :] >= 0]
            Xsums[a, b] = np.sum(X[clusterIndsa, :][:, clusterIndsb])
    Xsums = Xsums + Xsums.T
    for a in range(numClusters):  # relies on A begin symmetric and with no self-loops
        clusterIndsa = clusterInds[a, clusterInds[a, :] >= 0]
        Xsums[a, a] = np.sum(X[clusterIndsa, :][:, clusterIndsa]) / 2

    return Xsums


# works
def _fastNormalizedBMLogLik(thetaVec: np.ndarray, habSqrdVec: np.ndarray, sampleSize: int) -> float:
    thetaVec = bound_away_from_one_and_zero_arrays([thetaVec.astype(float)])[0]
    negEntVec = thetaVec * np.log(thetaVec) + (1 - thetaVec) * np.log(1 - thetaVec)
    normLogLik = np.sum(habSqrdVec * negEntVec) / sampleSize
    return normLogLik
