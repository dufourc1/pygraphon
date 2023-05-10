"""Estimator based on method of moments for cycle counts."""
from collections.abc import Iterable as IterableCollection
from itertools import product
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
from scipy.optimize import fsolve

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons.StepGraphon import StepGraphon
from pygraphon.subgraph_isomorphism.CycleCount import CycleCount
from pygraphon.utils.utils_graph import edge_density
from pygraphon.utils.utils_matrix import check_symmetric


class SimpleMomentEstimator(BaseEstimator):
    """Moment estimator uses subgraph isomorphims density to fit a step graphon.

    Suppose the block model has the same off-diagonal parameters and equally-sized blocks.
    """

    def __init__(
        self,
        blocks: Union[int, Iterable[float]],
    ) -> None:
        """Moment estimator uses subgraph isomorphims density to fit a step graphon.

        Parameters
        ----------
        blocks : Union[int, Iterable[float]]
            number of blocks or size of blocks

        Raises
        ------
        ValueError
            if blocks is not an integer or an iterable of floats
        ValueError
            if blocks is an iterable of floats and the sum is not 1
        ValueError
            blocks is not an integer or an iterable of floats
        ValueError
            if too many blocks are specified (can do between 2 and 8)
        """
        super().__init__()

        if type(blocks) in [int, float]:
            if int(blocks) != blocks:
                raise ValueError("invalid type for blocks, expecting an integer or a list of float")
            blocks = np.repeat(1 / blocks, blocks)
        elif isinstance(blocks, IterableCollection) and not isinstance(blocks, str):
            if np.sum(blocks) != 1:
                raise ValueError(f"Block sizes should add to one, but got {np.sum(blocks)}")
        else:
            raise ValueError(
                f"Blocks argument should be either the number of blocks, or a list of size of blocks, but got {blocks}"
            )
        self.blocks = blocks
        self.numberBlocks = len(blocks)

        self.numberParameters = self.numberBlocks + 1

        if self.numberParameters < 2 or self.numberParameters > 9:
            raise ValueError(
                f"block model can't have < 2 parameters or > 9 for now, but got {self.numberParameters}"
            )

        # count cycles of length 3,..,self.numberParameters-1
        self.counter = CycleCount(2 + self.numberParameters - 1)

    def _approximate_graphon_from_adjacency(
        self, adjacency_matrix: np.ndarray
    ) -> Tuple[StepGraphon, None]:
        """Estimate the graphon function f(x,y) from an adjacency matrix by solving moment equations.

        The P matrix is not returned as it is not used in the estimation.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjacency matrix

        Returns
        -------
        StepGraphon, None
            approximated graphon and None
        """
        # compute densities from observed graphs
        cycles = self._count_cycles(adjacency_matrix)
        rho = edge_density(adjacency_matrix)
        # solve the moment equations
        root = fsolve(
            func=self._get_moment_equations(cycles, rho),
            x0=np.array([rho ** (i + 1) for i in range(self.numberParameters)]),
        )
        # structure the parameters into a graphon
        graphon = self._add_constraints_on_SBM(root, self.numberBlocks)
        graphon = self.correct_fitted_values(graphon, kind="abs")
        return (
            StepGraphon(graphon, 1 / self.numberBlocks, initial_rho=rho),
            None,
        )

    @staticmethod
    def correct_fitted_values(graphon, kind="abs") -> np.ndarray:
        """Project the method of moment into the graphon space.

        Parameters
        ----------
        graphon : np.ndarray
             estimated graphon
        kind : str
            method of projection. Either absolute value ("abs") or clipping ("clip").
            Defaults to "abs".

        Returns
        -------
        np.ndarray
            projected graphon

        Raises
        ------
        ValueError
            if kind is not in ["abs", "clip"]
        """
        if kind == "abs":
            return np.abs(graphon)
        if kind == "clip":
            return np.clip(graphon, min=0)
        raise ValueError("kind should be either abs or clip")

    @staticmethod
    def _cycle_moments_theoretical(L: int, theta: np.ndarray, areas: np.ndarray = None) -> float:
        """Return the theoretical values of homomorphism densities of cycle of length L.

        Take as input a block model represented by a connection matrix theta and sizes of the blocks areas.

        Parameters
        ----------
        L : int
            length of the cycle density to compute: C_L
        theta : np.ndarray
            connection matrix. Should be in [0,1]^KxK and symmetric
        areas : np.ndarray
            array of size of blocks. Should be in [0,1]^K, symmetric and summing to one.
            Defaults to None, which sets all blocks to be of same size.

        Returns
        -------
        float
            t(C_L,W)

        Raises
        ------
        ValueError
            if theta is not symmetric
        ValueError
            if theta and the number of blocks are not compatible
        ValueError
            if the sum of the areas is not 1
        ValueError
            if the areas are not in [0,1]
        """
        if not check_symmetric(theta):
            raise ValueError("connection matrix theta should be symmetric")

        K = theta.shape[0]
        if areas is None:
            areas = np.ones(K) / K
        else:
            if len(areas) != K:
                raise ValueError(
                    f"connection matrix {K}x{K} and size of blocks {len(areas)} don't agree"
                )
            if np.max(areas) > 1 or np.min(areas) < 0:
                raise ValueError("areas should be in [0,1]")
            if np.sum(areas) != 1:
                raise ValueError(f"areas should sum up to one, not {np.sum(areas)}")

        result = 0

        for indices in list(product(range(K), repeat=L)):
            inter = (
                np.prod([theta[indices[i]][indices[i + 1]] for i in range(len(indices) - 1)])
                * theta[indices[-1]][indices[0]]
            )

            inter *= np.prod([areas[i] for i in indices])
            result += inter

        return result

    @staticmethod
    def _edge_density_moment_theoretical(theta: np.ndarray, areas: np.ndarray = None) -> float:
        """Return the theoretical values of homomorphism densities of edge density given a block model.

        Parameters
        ----------
        theta : np.ndarray
             connection matrix
        areas : np.ndarray
            sizes of the blocks. Defaults to None and then supposed to be homogeneous.

        Returns
        -------
        float
             theoretical edge density of the SBM represented by theta

        Raises
        ------
        ValueError
            if theta is not symmetric
        ValueError
            if theta and the number of blocks are not compatible
        ValueError
            if the sum of the areas is not 1
        ValueError
            if the areas are not in [0,1]
        """
        if not check_symmetric(theta):
            raise ValueError("connection matrix theta should be symmetric")

        K = theta.shape[0]
        if areas is None:
            areas = np.ones(K) / K
        else:
            if len(areas) != K:
                raise ValueError(
                    f"connection matrix {K}x{K} and size of blocks {len(areas)} don't agree"
                )
            if np.max(areas) > 1 or np.min(areas) < 0:
                raise ValueError("areas should be in [0,1]")
            if np.sum(areas) != 1:
                raise ValueError(f"areas should sum up to one, not {np.sum(areas)}")

        result = 0
        for i, j in list(product(range(K), repeat=2)):
            result += theta[i][j]
        return result / K**2

    @staticmethod
    def _cherry_density_moment_theoretical(theta: np.ndarray, areas: np.ndarray = None) -> float:
        """Return the theoretical values of homomorphism densities of cherry density given a block model.

        Parameters
        ----------
        theta : np.ndarray
            connection matrix
        areas : np.ndarray
            sizes of the blocks. Defaults to None and then supposed to be homogeneous.

        Returns
        -------
        float
            theoretical cherry density of the SBM represented by theta

        Raises
        ------
        ValueError
            if theta is not symmetric
        ValueError
            if theta and the number of blocks are not compatible
        ValueError
            if the sum of the areas is not 1
        ValueError
            if the areas are not in [0,1]
        """
        if not check_symmetric(theta):
            raise ValueError("connection matrix theta should be symmetric")

        K = theta.shape[0]
        if areas is None:
            areas = np.ones(K) / K
        else:
            if len(areas) != K:
                raise ValueError(
                    f"connection matrix {K}x{K} and size of blocks {len(areas)} don't agree"
                )
            if np.max(areas) > 1 or np.min(areas) < 0:
                raise ValueError("areas should be in [0,1]")
            if np.sum(areas) != 1:
                raise ValueError(f"areas should sum up to one, not {np.sum(areas)}")

        result = 0

        for indices in list(product(range(K), repeat=3)):
            inter = np.prod([theta[indices[i]][indices[i + 1]] for i in range(len(indices) - 1)])

            inter *= np.prod([areas[i] for i in indices])
            result += inter

        return result

    def _get_moment_equations(
        self,
        cyclesCounts: Iterable[float],
        edgeDensity: float,
    ) -> Callable:
        """Return the system of equations to solve to find the parameters of a block model (p,q).

        Based  on the moments (edge density and cycle densities)

        Parameters
        ----------
        cyclesCounts : Iterable[float]
            cycle (homomorphism) densities starting from C_3
        edgeDensity : float
            edge density (homomorphism density of K_2)

        Returns
        -------
        Callable
            system of equation to solve to pass to scipy.optimize
        """
        K = self.numberBlocks

        def func(x):
            if len(x) != self.numberParameters:
                raise ValueError(
                    f"length of parameter vector {len(x)} does not match expected number {self.numberParameters}"
                )

            theta = self._add_constraints_on_SBM(
                x, K
            )  # x[-1] * np.ones((K, K)) + (x[0:-1] - x[-1]) * np.eye(K)

            functions = [
                self._edge_density_moment_theoretical(theta) - edgeDensity,
            ]
            for L in range(self.numberParameters - 1):
                functions.append(self._cycle_moments_theoretical(L + 3, theta) - cyclesCounts[L])
            return functions

        return func

    @staticmethod
    def _add_constraints_on_SBM(x, K) -> np.ndarray:
        """Return a structured array with the constraints on the SBM parameters.

        Parameters
        ----------
        x : np.ndarray
             parameters of the SBM
        K : int
            number of blocks

        Returns
        -------
        np.ndarray
            constrained matrix of connectivity of the SBM
        """
        return x[-1] * np.ones((K, K)) + (x[0:-1] - x[-1]) * np.eye(K)

    def _count_cycles(self, instance: np.ndarray) -> List[float]:
        """Return the normalized count of cycle of length L in the instance of a simple graph.

        hom(C_k,G)/n^k for k in 3,..,9

        Parameters
        ----------
        instance : np.ndarray
            graph

        Returns
        -------
        List[float]
             homomorphism densities of cycle in instance
        """
        return self.counter(instance)


class MomentEstimator(SimpleMomentEstimator):
    """Estimate the moments of a block model using the moment equations.

    Does not assume specific structure on the blockmodel fitted apart from homogeneous block sizes.
    """

    def __init__(self, blocks: Union[int, Iterable[float]]) -> None:
        super().__init__(blocks)
        self.numberParameters = self.numberBlocks * (self.numberBlocks - 1) // 2 + self.numberBlocks
        if self.numberParameters > 9:
            raise ValueError("number of parameters should be <= 9")
        self.counter = CycleCount(self.numberParameters - 1)

    def _add_constraints_on_SBM(self, x, K) -> np.ndarray:
        # build theta based on x
        theta = np.zeros((K, K))
        theta += np.diag(x[0:K])
        index = 0
        for i in range(0, K):
            for j in range(i + 1, K):
                theta[i][j] = x[K + index]
                theta[j][i] = x[K + index]
                index += 1
        return theta
