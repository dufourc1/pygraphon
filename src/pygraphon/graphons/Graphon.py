"""Graphon based on a continuous function."""
import math
import warnings
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from scipy.integrate import dblquad

from pygraphon.utils.utils_func import copy_func


class Graphon:
    r"""Basic Graphon defined only by its function :math:`f(x,y)`.

    If scaled, will initialize a scaled graphon with integral equal to 1.
    The initial_rho parameter is used to keep track of the original edge density of the graphon.
    If additionaly check is true, will check if the graphon integrates to 1 and if not, will try to correct it.

    Parameters
    ----------
    function : Callable
        graphon function :math:`f:[0,1]^2 → [0,1]`
    scaled : bool
        Should the graphon be scaled (integrate to 1), by default True
    check : bool
        Should we try to enforce scaled graphon, by default True
    initial_rho : Optional[float]
        initial edge density, by default None. Should be in ]0,1[ if given. The function is
        expected to integrate to 1 if scaled is True and initial_rho is not None.


    ..  note::
        Note on rescaled graphons and the current implementation.

        Internally, the graphon is always rescaled to have integral equal to 1 and be indentifiable.
        However, when sampling from the graphon, the original edge density is used, meaning we generate an edge with
        :math:`\operatorname{Bern}(\rho f(x,y))`, where :obj:`rho` is the user given parameter in  :py:meth:`draw`,
        and :math:`f(\cdot,\cdot)` is :obj:`function` given at initialization.
        to the init function.

        if :py:obj:`initial_rho` is given but the function does not integrate to 1, the integral of the function is used
        instead of :py:obj:`initial_rho` for consistency.
    """

    def __init__(
        self,
        function: Callable,
        initial_rho: Optional[float] = None,
        scaled=True,
        check=True,
    ) -> None:
        self.graphon_function = function
        self.scaled = scaled
        self.integral_value = deepcopy(self.integral())

        # remember the original edge density of the graphon
        if initial_rho is None:
            self.initial_rho = deepcopy(self.integral_value)
        # sanitiy check on initial_rho
        else:
            if initial_rho < 0 or initial_rho > 1:
                raise ValueError("Initial edge density must be in  ]0,1[")

            if scaled and not math.isclose(self.integral_value, 1):
                warnings.warn(
                    "function provided does not integrate to 1, disregarding initial_rho, replacing with "
                    + "integral_value of given function",
                    UserWarning,
                    stacklevel=2,
                )
                self.initial_rho = deepcopy(self.integral_value)
            else:
                self.initial_rho = deepcopy(initial_rho)

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

    def check_graphon_integral(self):
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

    def draw(self, n: int, exchangeable: bool = True, rho: Optional[float] = None) -> np.ndarray:
        """Draw a graph from the graphon with a given density and number of vertices.

        Parameters
        ----------
        n : int
             number of vertices of the realized graph
        exchangeable : bool
            if True, the graph generated is exchangeable. Defaults to True.
        rho : Optional[float]
            edge density of the realized graph (if :py:obj:`None`, will use draw from the unormalized graphon
            with density :py:attr:`initial_rho`)

        Returns
        -------
        np.ndarray
            adjacency matrix of the realized graph (nxn)


        Raises
        ------
        ValueError
            if :py:obj:`rho` is not in :math:`]0,1[`
        """
        probs = self.get_edge_probabilities(n, exchangeable=exchangeable, wholeMatrix=False)
        if rho is None:
            scale = 1
        else:
            if rho < 0 or rho > 1:
                raise ValueError("Edge density must be in  ]0,1[")
            scale = rho / self.initial_rho
        return self._generate_adjacency_matrix(n, probs, scale)

    @staticmethod
    def _generate_adjacency_matrix(n, probs, rho):
        """Generate adjacency matrix :math:`A_{ij} = 1` with probability :attr:`rho*probs_ij`.

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

    def _get_edge_probabilities(self, n, latentVarArray, wholeMatrix=True):
        """Generate a matrix P_ij with  0 =< i,j <= n-1.

        Parameters
        ----------
        n : int
            number of nodes in the edge probability matrix returned
        latentVarArray : np.ndarray
            array of latent variables (n) used to generate the edge probabilities

        Returns
        -------
        np.ndarray
            matrix of edge probabilities (nxn)
        """
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

    def get_edge_probabilities(
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

        return self._get_edge_probabilities(n, latentVarArray, wholeMatrix)

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
