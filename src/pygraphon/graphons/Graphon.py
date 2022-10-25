"""Graphon based on a continuous function."""
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from scipy.integrate import dblquad

from pygraphon.utils.utils_func import copy_func


class Graphon:
    """Basic Graphon defined only by its function f(x,y).

    If scaled, will initialize a scaled graphon with integral equal to 1.
    The initial_rho parameter is used to keep track of the original edge density of the graphon.
    If additionaly check is true, will check if the graphon integrates to 1 and if not, will try to correct it.

    Parameters
    ----------
    function : Callable
        graphon function f(x,y)
    scaled : bool
        Should the graphon be scaled (integrate to 1), by default True
    check : bool
        Should we try to enforce scaled graphon, by default True
    initial_rho : Optional[float]
        initial edge density, by default None
    """

    def __init__(
        self,
        function: Callable,
        initial_rho: Optional[float] = None,
        scaled=True,
        check=True,
    ) -> None:
        """Initialize a graphon defined only by its function f(x,y).

        If scaled, will initialize a scaled graphon with integral equal to 1.
        The initial_rho parameter is used to keep track of the original edge density of the graphon.
        If additionaly check is true, will check if the graphon integrates to 1 and if not, will try to correct it.

        Parameters
        ----------
        function : Callable
            graphon function f(x,y)
        scaled : bool
            Should the graphon be scaled (integrate to 1), by default True
        check : bool
            Should we try to enforce scaled graphon, by default True
        initial_rho : Optional[float]
            initial edge density, by default None

        Raises
        ------
        ValueError
            if graphon does not integrate to 1 and cannot be automatically scaled
        """
        self.integral_value = None
        self.graphon_function = function
        self.scaled = scaled

        # remember the original edge density of the graphon
        if initial_rho is None:
            self.initial_rho = deepcopy(self.integral())
        else:
            self.initial_rho = deepcopy(initial_rho)

        #
        if check:
            self.check_graphon()
            if not self.check_graphon_integral() and self.scaled:
                try:
                    self.correct_graphon_integral()
                except NotImplementedError:
                    raise ValueError(
                        "Graphon does not integrate to 1 and cannot be automatically corrected"
                    )

    def check_graphon(self):
        """Check graphon properties depending on the subclass that iplements it.

        Raises
        ------
        ValueError
            if the graphon function is not callable
        """
        if not callable(self.graphon_function):
            raise ValueError("Graphon function must be callable")

    def correct_graphon_integral(self):
        """Normalize the graphon function to 1 if possible.

        Raises
        ------
        ValueError
             if graphon integrates to 0.
        """
        integral = self.integral()
        if integral == 0:
            raise ValueError("graphon integrates to 0, cannot be automatically corrected")
        # keep copy of old graphon function to be able to rescale it
        # without having infinite recursion
        self.old_func = copy_func(self.graphon_function)
        self.graphon_function = lambda x, y: self.old_func(x, y) / integral

    def check_graphon_integral(self) -> bool:
        """Check if the graphon integrates to 1.

        Returns
        -------
        bool
            True if graphon integrates to 1, False otherwise
        """
        return self.integral() == 1

    def integral(self) -> float:
        """Compute the integral of the graphon.

        Returns
        -------
        float
            integral value of the graphon
        """
        integral = 2 * dblquad(self.graphon_function, 0, 1, lambda x: 0, lambda x: x)[0]
        return integral

    def draw(self, rho: float, n: int, exchangeable: bool = True) -> np.ndarray:
        """Draw a graph from the graphon with a given density and number of vertices.

        Parameters
        ----------
        rho : float
            edge density of the realized graph
        n : int
             number of vertices of the realized graph
        exchangeable : bool
            if True, the graph generated is exchangeable. Defaults to True.

        Returns
        -------
        np.ndarray
            adjacency matrix of the realized graph (nxn)
        """
        probs = self._get_edge_probabilities(n, exchangeable=exchangeable, wholeMatrix=False)
        return self._generate_adjacency_matrix(n, probs, rho)

    @staticmethod
    def _generate_adjacency_matrix(n, probs, rho):
        """Generate adjacency matrix A_ij = 1 with probability rho*probs_ij.

        Parameters
        ----------
        n: int
            number of nodes in the graph (A will be a nxn matrix)
        probs: np.ndarray or int
            bernoulli probability of having an edge if np.ndarray size n*(n-1)/2 with indices corresponding
            to np.tri_indices(n,1)
            if int : constant probability for all the edges
        rho: float
            edge density of the realized graph

        Returns
        -------
        np.array
            adjacency matrix ind realisations of Bern(probs) (nxn)

        Raises
        ------
        ValueError
            if  probs and n do not agree on size
        """
        if not isinstance(probs, int):
            if probs.shape[0] != int(n * (n - 1) / 2):
                raise ValueError(
                    f"probs array wrong size compared to number of nodes: got {probs.shape} instead of { n*(n-1)/2}"
                )

        # generate bernoulli draws with help of uniform rv
        r = (np.random.uniform(size=int(n * (n - 1) / 2)) < rho * probs) * 1

        a = np.zeros((n, n))

        # fiil triangles of the matrix
        a[np.triu_indices(n, 1)] = r
        a[np.tril_indices(n, -1)] = a.T[np.tril_indices(n, -1)]
        return a

    def _get_edge_probabilities(
        self,
        n: int,
        exchangeable: bool = True,
        wholeMatrix: bool = True,
    ) -> np.ndarray:
        """Generate a matrix P_ij with  0 =< i,j <= n-1.

        Parameters
        ----------
        n : int
            number of nodes in the edge probability matrix returned
        exchangeable : bool
            if True the graph will be vertex exchangeable. Defaults to True.
        wholeMatrix : bool
             if True return the square symmetric matrix, otherwise return the upper
            diagonal. Defaults to True.

        Returns
        -------
        np.ndarray
            matrix of edge probabilities (nxn) if wholeMatrix (n*(n-1)/2) if not wholeMatrix
        """
        latentVarArray = (
            np.random.uniform(0, 1, size=n) if exchangeable else np.array([i / n for i in range(n)])
        )

        # generate edge probabilities from latent variables array
        probs = np.zeros((int(n * (n - 1) / 2), 1)).reshape(-1)

        # TODO: performance
        # Now we iterate over the probabilities array and call the function for each
        # latent variable. Way to do that in a vectorized fashion to avoid for
        # loop ?
        I, J = np.triu_indices(n, 1)
        for index, nodes in enumerate(zip(I, J)):
            probs[index] = self.graphon_function(latentVarArray[nodes[0]], latentVarArray[nodes[1]])
        probs *= self.initial_rho

        if wholeMatrix:
            P = np.zeros((n, n))
            P[np.triu_indices(n, 1)] = probs
            P[np.tril_indices(n, -1)] = P.T[np.tril_indices(n, -1)]
            return P
        return probs

    def __add__(self, other):
        """Add and normalize two graphons.

        Parameters
        ----------
        other : Graphon
             other graphon to add.

        Returns
        -------
        Graphon
            graphon f1(x,y) + f2(x,y) (normalized to 1)
        """

        def _function(x, y):
            return 0.5 * self.graphon_function(x, y) + 0.5 * other.graphon_function(x, y)

        return Graphon(_function, check=False)
