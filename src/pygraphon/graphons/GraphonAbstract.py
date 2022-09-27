from abc import ABC, abstractclassmethod
from copy import deepcopy
from typing import Callable

import numpy as np


class GraphonAbstract(ABC):
    """Abstract class for Graphon.
    All graphons of this class will be scaled graphon, meaning the integral of f(x,y) over [0,1]^2 is 1.
    """

    def __init__(self, initial_rho=None, scaled=True, check=True) -> None:
        """Constructor for Graphon.

        Will check that graphon is correctly build.
        """
        super().__init__()

        # remember the original edge density of the graphon
        if initial_rho is None:
            self.initial_rho = deepcopy(self.integral())
        else:
            self.initial_rho = deepcopy(initial_rho)

        if check:
            self.check_graphon()
            if not self.check_graphon_integral() and scaled:
                try:
                    self.correct_graphon_integral()
                except NotImplementedError:
                    raise ValueError(
                        "Graphon does not integrate to 1 and cannot be automatically corrected"
                    )
        self.graphon_function = self.graphon_function_builder()

    @abstractclassmethod
    def graphon_function_builder(self) -> Callable:
        """
        Build the graphon function f(x,y)
        """

    def check_graphon(self):
        """
        Check graphon properties depending on the subclass that iplements it.
        Only implemented in subclasses that need to impose certain properties.

        Raises:
            ValueError: if graphon does not integrate to 1 it it cannot be automatically scaled
        """

    def check_graphon_integral(self) -> bool:
        """Check if the graphon integrates to 1.

        Args:
            n (int): number of nodes in the graphon
            exchangeable (bool, optional): if True the graph will be vertex exchangeable. Defaults to True.

        Returns:
            bool: True if graphon integrates to 1, False otherwise
        """
        return self.integral() == 1

    def correct_graphon_integral(self):
        """
        Correct the integral of the graphon function f(x,y) to 1
        """
        raise NotImplementedError

    @abstractclassmethod
    def integral(self):
        """
        Return the integral of the graphon function f(x,y) over [0,1]^2

        Returns:
            float: integral of the graphon function
        """

    def draw(self, rho: float, n: int, exchangeable: bool = True) -> np.ndarray:
        """Draw a graph from the graphon with a given density and number of vertices.

        Args:
            rho (float): edge density of the realized graph
            n (int): number of vertices of the realized graph
            exchangeable (bool, optional): if True, the graph is exchangeable. Defaults to True.

        Returns:
            np.ndarray: adjacency matrix of the realized graph (nxn)
        """
        probs = self._get_edge_probabilities(n, exchangeable=exchangeable, wholeMatrix=False)
        return self._generate_adjacency_matrix(n, probs, rho)

    @staticmethod
    def _generate_adjacency_matrix(n, probs, rho):
        """
        Generate adjacency matrix A_ij = 1 with probability rho*probs_ij

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
        """Generate a matrix P_ij with  0 =< i,j <= n-1

        Args:
            n (int): number of nodes in the edge probability matrix returned
            exchangeable (bool, optional): if True the graph will be vertex exchangeable. Defaults to True.
            wholeMatrix (bool, optional): if True return the square symmetric matrix, otherwise return the upper
            diagonal. Defaults to True.

        Returns:
            np.ndarray: matrix of edge probabilities (nxn) if wholeMatrix (n*(n-1)/2) if not wholeMatrix
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
        """
        Overload the + operator to add two graphons
        """
        if not isinstance(other, GraphonAbstract):
            raise TypeError(f"Can only add two graphons, got {type(other)} instead")

        raise NotImplementedError
