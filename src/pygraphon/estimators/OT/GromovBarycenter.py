from typing import Iterable, Tuple, Union

import networkx as nx
import numpy as np
from ot.gromov import entropic_gromov_barycenters

from pygraphon.estimators.BaseEstimator import BaseEstimator
from pygraphon.graphons import Graphon, StepGraphon


class GromovBarycenter(BaseEstimator):

    def _approximate_graphon_from_adjacency(
        self,
        adjacency_matrix: np.ndarray,
        k,
        max_iter=10,
        type: str = "argmax",
        **kwargs,
    ) -> Tuple[Graphon, np.ndarray]:
        """Estimate the graphon function f(x,y) values from a realized graph adjacency matrix.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            adjacency matrix of a simple graph
        k : int
            number of clusters for the Gromov-Wasserstein barycenter
        type : str
            type of conversion from transport map to estimated edge connectivity matrix, by default "argmax".
        kwargs: optional
            additional arguments to pass to optimal transport solver

        Returns
        -------
        Tuple[Graphon, np.ndarray]
            estimated graphon function as a Graphon object and estimated edge connectivity matrix
        """

        graphon, Ps = self.fit_multiple(
            [adjacency_matrix], k, max_iter=max_iter, type=type, **kwargs
        )
        return graphon, Ps[0]

    def barycenter_to_P_hat(
        self,
        transport_map: np.ndarray,
        barycenter: np.ndarray,
        type: str = "argmax",
    ) -> np.ndarray:
        """Convert the transport map to the estimated edge connectivity matrix P_hat.

        Parameters
        ----------
        transport_map : np.ndarray
            transport map from the Gromov-Wasserstein barycenter computation
            dimension k x n where k is the size of the barycenter and n the size of the graph
        barycenter : np.ndarray
            barycenter graphon matrix of size k x k
        type : str
            type of conversion, by default "argmax". Can be "argmax" or "weighted"

        Returns
        -------
        np.ndarray
            estimated edge connectivity matrix P_hat
        """
        n = transport_map.shape[1]
        if type == "weighted":
            P_hat = np.transpose(transport_map) @ barycenter @ transport_map * n**2
        elif type == "argmax":
            indices = np.argmax(transport_map, axis=0)
            P_hat = barycenter[indices][:, indices]
        else:
            raise ValueError("type must be 'argmax' or 'weighted'")
        return P_hat

    def fit_multiple(
        self,
        graphs: Union[Iterable[nx.Graph], Iterable[np.ndarray]],
        k: int,
        max_iter: int = 10,
        type: str = "argmax",
        **kwargs,
    ):
        self.fitted = True
        if isinstance(graphs[0], nx.Graph):
            graphs = [nx.to_numpy_array(graph) for graph in graphs]

        res = entropic_gromov_barycenters(
            k,
            graphs,
            p=None,
            log=True,
            symmetric=True,
            max_iter=max_iter,
            **kwargs,
        )
        theta = res[0]
        Ps = [
            self.barycenter_to_P_hat(res[1]["T"][i], theta, type=type) for i in range(len(graphs))
        ]
        return StepGraphon(theta, bandwidthHist=1 / k), Ps
