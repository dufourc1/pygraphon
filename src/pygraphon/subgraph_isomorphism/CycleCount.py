"""Class to compute cycle homomorphism."""
import numpy as np

from pygraphon.utils.utils_graph import check_simple_adjacency_matrix


class CycleCount:
    """Implement the cycle count algorithm for finding the number of cycles in a graph.

    The algorithm is based on the paper:
    """

    def __init__(self, L: float = 9) -> None:
        if L < 3:
            raise ValueError("input L should be an integer >= 3")
        if L >= 10:
            raise ValueError("cycleCount algorithm cannot handle L > 10")

        self.L = float(L)
        self.counts = None

    def __call__(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Count the densities of subgraph C_l in a graph G: t(C_L,G).

        Parameters
        ----------
        adjacency_matrix : np.ndarray
             Adjacency matrix representing the simple graph G

        Returns
        -------
        np.ndarray
            counts of cycle of length 3 to  L
        """
        check_simple_adjacency_matrix(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix
        return self.network_profile(adjacency_matrix, kmax=self.L)

    def approximate_network_profile(
        self,
        adjacency_matrix: np.ndarray,
        kmax: int,
        subsample: float = 0.4,
        repetitions: int = 3,
    ) -> np.ndarray:
        """Compute the network profile of a graph G by subsampling it multiple times and averaging the results.

        Will compute the different network profile in parallel

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Adjacency matrix representing the simple graph G
        kmax : int
            maximum length of cycles to consider (default: None). If None, use the value of L.
        subsample : float
            fraction of the graph to subsample (default: 0.4)
        repetitions : int
             number of times to subsample the graph (default: 3)

        Raises
        ------
        NotImplementedError
            NotImplemented for now
        """
        raise NotImplementedError("approximate_network_profile is not implemented")

    def network_profile(self, adjacency_matrix, kmax=None) -> np.ndarray:
        """Compute the network profile of a graph: normalized counts of cycles of length 3 to L.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
             Adjacency matrix representing the simple graph G
        kmax : int
            maximum length of cycles to consider (default: None). If None, use the value of L.

        Returns
        -------
        np.ndarray
            network profile of the graph

        Raises
        ------
        NotImplementedError
            if L is bigger than 10
        ValueError
            if L is smaller than 3
        """
        if kmax is None:
            kmax = self.L
        if kmax >= 10:
            raise NotImplementedError(
                "Cannot count cycles with length >= 10:  not implemented. Please change the value of kmax."
            )
        kmax = int(kmax)
        number_edges = np.sum(adjacency_matrix) / 2
        degrees = np.sum(adjacency_matrix, axis=0)
        non_zero_degrees = np.nonzero(degrees)[0]
        A1 = adjacency_matrix[non_zero_degrees, :][:, non_zero_degrees]
        A2 = A1 @ A1
        dA2 = np.diag(A2)
        Ak = A2
        t = np.zeros(kmax)
        for k in range(3, kmax + 1):
            Ak = Ak @ A1
            if k == 3:
                dA3 = np.diag(Ak)
                tA3 = np.sum(dA3)
                sA3 = np.sum(Ak)
                t[k - 1] = tA3 ** (1 / k)
                A3 = Ak
            elif k == 4:
                dA4 = np.diag(Ak)
                tA4 = np.sum(dA4)
                t[k - 1] = (
                    tA4
                    + 2 * number_edges
                    - 2 * (degrees[non_zero_degrees] @ degrees[non_zero_degrees])
                ) ** (1 / k)
                A4 = Ak

            elif k == 5:
                dA5 = np.diag(Ak)
                tA5 = np.sum(dA5)
                t[k - 1] = (tA5 - 5 * np.sum((degrees[non_zero_degrees] - 1) * dA3)) ** (1 / k)
                A5 = Ak
            elif k == 6:
                dA6 = np.diag(Ak)
                tA6 = np.sum(dA6)
                inter = tA6 - 3 * np.sum(dA3**2)
                inter += 9 * np.sum((A2**2) * A1)
                inter -= 6 * np.sum(dA4 * (degrees[non_zero_degrees] - 1))
                inter -= 4 * np.sum(dA3 - degrees[non_zero_degrees] ** 3)
                inter += 3 * sA3
                inter -= 12 * np.sum(degrees[non_zero_degrees] ** 2)
                inter += 4 * np.sum(degrees[non_zero_degrees])
                t[k - 1] = (inter) ** (1 / k)
            elif k == 7:
                dA7 = np.diag(Ak)
                tA7 = np.sum(dA7)
                inter = tA7 - 7 * np.sum(dA3 * dA4) + 7 * np.sum((A2**3) * A1)
                inter -= 7 * np.sum(dA5 * degrees[non_zero_degrees])
                inter += 21 * np.sum(A3 * A2 * A1)
                inter += 7 * tA5
                inter -= 28 * np.sum((A2**2) * A1)
                inter += 7 * np.sum(
                    A2 * A1 * np.outer(degrees[non_zero_degrees], (degrees[non_zero_degrees]))
                )
                inter += 14 * np.sum(dA3 * degrees[non_zero_degrees] ** 2)
                inter += 7 * np.sum(dA3 * np.sum(A2, axis=1))
                inter -= 77 * np.sum(dA3 * degrees[non_zero_degrees])
                inter += 56 * tA3
                t[k - 1] = (inter) ** (1 / k)
            elif k == 8:
                inter = (
                    np.sum(np.diag(Ak))
                    - 4 * np.sum(dA4 * dA4)
                    - 8 * np.sum(dA3 * dA5)
                    - 8 * np.sum(dA2 * dA6)
                    + 16 * np.sum(dA2 * dA3 * dA3)
                    + 8 * np.sum(dA6)
                    + 16 * np.sum(dA4 * dA2 * dA2)
                    - 72 * np.sum(dA3 * dA3)
                    - 96 * np.sum(dA4 * dA2)
                    - 12 * np.sum(dA2 * dA2 * dA2 * dA2)
                    + 64 * np.sum(dA2 * dA3)
                    + 73 * np.sum(dA4)
                    + 72 * np.sum(dA2 * dA2 * dA2)
                    - 112 * np.sum(dA3)
                    + 36 * np.sum(dA2)
                )
                inter += (
                    2 * np.sum(A2 * A2 * A2 * A2)
                    + 24 * np.sum(A2 * A2 * A1 * A3)
                    + 4 * np.sum(np.outer(dA3, dA3) * A1)
                    + 16 * np.sum(np.outer(dA2, dA3) * A1 * A2)
                    + 12 * np.sum(A1 * A3 * A3)
                    + 24 * np.sum(A1 * A4 * A2)
                    + 4 * np.sum(np.outer(dA2, dA2) * A2 * A2)
                    + 8 * np.sum(A2 @ np.diag(dA4))
                    + 8 * np.sum(np.outer(dA2, dA2) * A1 * A3)
                    - 16 * np.sum(A2 * A2 * A2)
                    - 32 * np.sum(A1 * A2 * A3)
                    - 96 * np.sum(np.diag(dA2) * (A2 * A2 * A1))
                    - 4 * np.sum(A4)
                    - 16 * np.sum(np.diag(dA2 * dA2) @ A2)
                    + 272 * np.sum(A2 * A2 * A1)
                    + 48 * np.sum(A3)
                    - 132 * np.sum(A2)
                )
                inter += -64 * np.sum(A1 * ((A1 * A2) ** 2)) - 24 * np.sum(
                    A1 * (A1 @ np.diag(dA2) @ A1) * A2
                )
                xk4 = 0
                for i in range(A1.shape[0]):
                    xk4 += A1[i, :] @ (A1 * (A1 @ np.diag(A1[i, :]) @ A1)) @ (A1[i, :].T)
                inter += 22 * xk4
                t[k - 1] = (inter) ** (1 / k)
            elif k == 9:
                inter = (
                    np.sum(np.diag(Ak))
                    - 9 * np.sum(dA4 * dA5)
                    - 9 * np.sum(dA3 * dA6)
                    - 9 * np.sum(dA2 * dA7)
                    + 6 * np.sum(dA3 * dA3 * dA3)
                    + 36 * np.sum(dA4 * dA3 * dA2)
                    + 9 * np.sum(dA7)
                    + 18 * np.sum(dA2 * dA2 * dA5)
                    - 171 * np.sum(dA4 * dA3)
                    - 117 * np.sum(dA2 * dA5)
                    - 54 * np.sum(dA3 * dA2 * dA2 * dA2)
                    + 72 * np.sum(dA3 * dA3)
                    + 81 * np.sum(dA5)
                    + 504 * np.sum(dA3 * dA2 * dA2)
                    - 1746 * np.sum(dA3 * dA2)
                    + 1148 * np.sum(dA3)
                )
                inter = (
                    inter
                    + 9 * np.sum(A2 * A2 * A2 * A3)
                    + 9 * np.sum(np.outer(dA3, dA3) * A1 * A2)
                    + 27 * np.sum(A1 * A3 * A3 * A2)
                    + 27 * np.sum(A2 * A2 * A1 * A4)
                    + 9 * np.sum(np.outer(dA3, dA4) * A1)
                    + 9 * np.sum(np.outer(dA2, dA3) * A2 * A2)
                    + 18 * np.sum((np.outer(dA2, dA4) * A1 * A2))
                    + 18 * np.sum((np.outer(dA2, dA3) * A1 * A3))
                    + 27 * np.sum(A1 * A4 * A3)
                    + 27 * np.sum(A1 * A5 * A2)
                    + 9 * np.sum(np.outer(dA2, dA2) * A2 * A3)
                    + 9 * np.sum(A2 @ np.diag(dA5))
                    + 9 * np.sum((np.outer(dA2, dA2) * A4 * A1))
                    - 72 * np.sum(np.diag(dA2) @ (A2 * A2 * A2 * A1))
                    - 108 * np.sum(np.diag(dA3) @ (A2 * A2 * A1))
                    - 36 * np.sum(A2 * A2 * A3)
                    - 36 * np.sum(A4 * A1 * A2)
                    - 216 * np.sum(np.diag(dA2) @ (A1 * A3 * A2))
                    - 9 * np.sum(A3 @ np.diag(dA3))
                    - 36 * np.sum(np.diag(dA3 * dA2) @ A2)
                    - 18 * np.sum(np.outer(dA2**2, dA3) * A1)
                    - 36 * np.sum(np.outer(dA2**2, dA2) * A1 * A2)
                    + 336 * np.sum(A1 * A2 * A2 * A2)
                    + 288 * np.sum(np.diag(dA2) @ (A1 * A2 * A2))
                    + 684 * np.sum(A1 * A3 * A2)
                    + 171 * np.sum(A2 @ np.diag(dA3))
                    + 252 * np.sum((np.outer(dA2, dA2) * A1 * A2))
                    - 1296 * np.sum(A1 * A2 * A2)
                )
                inter = (
                    inter
                    - 48 * np.sum(A1 * A2 * ((A1 * A2) ** 2))
                    - 27 * np.sum((np.diag(dA2) @ A1) * (A1 @ (A1 * A2 * A2)))
                    - 72 * np.sum(A1 * ((A1 * A2) @ (A2 * A2)))
                    - 27 * np.sum(A1 * A2 * (A1 @ np.diag(dA3) @ A1))
                    - 144 * np.sum(A1 * ((A1 * A3) @ (A1 * A2)))
                    - 27 * np.sum(A1 * A3 * (A1 @ np.diag(dA2) @ A1))
                    - 54 * np.sum(A1 * A2 * (A2 @ np.diag(dA2) @ A1))
                    - 18
                    * np.sum(
                        (np.diag(dA2) @ (A1 * A2)) * np.tile(np.sum(A2, axis=0), (A2.shape[0], 1))
                    )
                    - 3 * np.sum((np.outer(dA2, dA2) * (A1 * (A1 @ np.diag(dA2) @ A1))))
                    + 324 * np.sum(A1 * A2 * ((A1 * A2) @ A1))
                    + 180 * np.sum((np.diag(dA2) @ A1) * (A1 @ (A1 * A2)))
                )
                for i1 in range(A1.shape[0]):
                    inter = inter + 99 * (
                        A1[i1, :] @ (A1 * (A1 @ np.diag(A2[i1, :]) @ A1)) @ (A1[i1, :].T)
                    )
                    inter = inter + 99 * (
                        A1[i1, :] @ (A1 * (A1 @ np.diag(A1[i1, :]) @ (A1 * A2))) @ (A1[i1, :].T)
                    )
                inter = inter - 156 * xk4
                t[k - 1] = (inter) ** (1 / k)
            else:
                raise ValueError("kmax should be <= 9 and >= 3")

        # normalize densities
        self.counts = np.copy((t ** (np.arange(0, len(t)) + 1))[2:])
        t /= np.shape(adjacency_matrix)[0]
        return (t ** (np.arange(0, len(t)) + 1))[2:]
