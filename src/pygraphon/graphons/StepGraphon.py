"""Stepgraphon class represent all stepfunction approximation of a continuous graphon."""
import math
from typing import Callable, Optional

import numpy as np

from pygraphon.graphons.Graphon import Graphon
from pygraphon.utils import check_symmetric, compute_areas_histogram


class StepGraphon(Graphon):
    """A step function graphon defined by giving the matrix representing the block model approximation.

    Parameters
    ----------
        graphon : np.ndarray
            np array representing the theta matrix
        bandwidthHist : float
            size of the groups (between 0 and 1).
        initial_rho : Optional[float]
            initial edge density (used to keep track in case of normalization), by default None
    """

    def __init__(
        self,
        graphon: np.ndarray,
        bandwidthHist: float,
        initial_rho: Optional[float] = None,
    ) -> None:
        # save args
        self.graphon = graphon
        if bandwidthHist > 1 or bandwidthHist <= 0:
            raise ValueError("The bandwidth should be between 0 and 1.")
        self.bandwidthHist = bandwidthHist
        if self.graphon.shape[0] != int(math.ceil(1 / self.bandwidthHist)):
            raise ValueError("The graphon matrix should have size consistent with the bandwidth.")

        if np.max(graphon) > 1 and initial_rho is None:
            raise ValueError(
                "The graphon matrix should be between 0 and 1 if no initial rho is given."
            )

        self.areas = compute_areas_histogram(self.graphon, self.bandwidthHist)
        self.remainder = 1 - int(1 / self.bandwidthHist) * self.bandwidthHist

        super().__init__(function=self.graphon_function_builder(), initial_rho=initial_rho)

        self.repr = "StepGraphon \n"
        self.repr += np.array2string(
            self.graphon * self.initial_rho,
            precision=3,
            floatmode="maxprec_equal",
        )
        self.repr += "\n\n "
        self.repr += np.array2string(self.areas[0, :], precision=3, floatmode="maxprec_equal")
        self.repr += " (size of the blocks)"

    def get_theta(self) -> np.ndarray:
        """Return the theta matrix in [0,1]^KxK.

        Returns
        -------
        np.ndarray
            theta matrix
        """
        return self.graphon * self.initial_rho

    def graphon_function_builder(self) -> Callable:
        """Build the graphon function f(x,y).

        Returns
        -------
        Callable
            graphon function
        """

        def _function(x: float, y: float) -> float:
            """Return the value of the graphon at the point (x,y).

            Parameters
            ----------
            x : float
                coordinate x (first latent variable)
            y : float
                coordinate y (second latent variable)

            Returns
            -------
            float
                f(x,y)
            """
            return self.graphon[int(x // self.bandwidthHist)][int(y // self.bandwidthHist)]

        return _function

    def correct_graphon_integral(self):
        """Normalize the graphon such that the integral is equal to 1 if needed."""
        self.normalize()

    def check_graphon(self):
        """Check if the graphon is symmetric, positive.

        Raises
        ------
        ValueError
             if the graphon is not symmetric
        ValueError
            if the graphon is not non negative
        """
        if not check_symmetric(self.graphon):
            raise ValueError("graphon matrix should be symmetric")
        if not np.all(self.graphon >= 0):
            raise ValueError("graphon matrix should be non-negative")

    def integral(self, graphon=None, areas=None) -> float:
        """Integrate the graphon over [0,1]x[0,1].

        Parameters
        ----------
        graphon : np.ndarray, optional
            theta matrix, by default None
        areas : np.ndarray, optional
            areas of the different blocks, by default None

        Returns
        -------
        float
            the value of the integral
        """
        if graphon is None:
            graphon = self.graphon
        if areas is None:
            areas = self.areas
        return np.sum(graphon * areas)

    def normalize(self) -> None:
        """Normalize graphon such that the integral is equal to 1.

        If the graphon is the empty graphon, does not do anything
        """
        integral = self.integral()
        if integral != 0:
            self.graphon = self.graphon / self.integral()
        else:
            self.graphon = self.graphon

    def get_graphon(self) -> np.ndarray:
        r"""Get the graphon matrix.

        Returns
        -------
        np.ndarray
            graphon connectivity matrix (\Theta)
        """
        return self.graphon

    def get_number_groups(self) -> int:
        """Return the number of groups of the graphon.

        Returns
        -------
        int
            number of groups
        """
        return int(1 // self.bandwidthHist) + 1
