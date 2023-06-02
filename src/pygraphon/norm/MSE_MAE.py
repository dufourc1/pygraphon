import numpy as np

from pygraphon.graphons import Graphon

from .BaseMetric import ValueMetric


class MseProbaEdge(ValueMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, graphon_2: Graphon, **kwargs) -> float:
        r"""Compute the MSE between the probability matrix of 2 different graphons.

        This follows the formula :math: `MSE = \frac{1}{n^2} \sum_{i,j} (p_{ij} - \hat{p}_{ij})^2` where
        :math: `p_{ij}` is the probability of an edge between nodes :math: `i` and :math: `j` in the actual
        graphon and :math: `\hat{p}_{ij}` is the probability of an edge between nodes :math: `i` and
        :math: `j` in the graphon_2 graphon. Can be used to compare a theoretical graphon and an estimated
        graphon.

        Parameters
        ----------
        graphon : Graphon
            The theoretical graphon to compare to
        graphon_2 : Graphon
            The graphon_2 graphon

        Returns
        -------
        float
            The MSE between the probability matrix of the graphon and the graphon_2 graphon

        Raises
        ------
        ValueError
            If the probability matrices have different shapes
        """
        p_0 = graphon.get_edge_probabilities(self.n_nodes, exchangeable=False, wholeMatrix=False)
        p_hat = graphon_2.get_edge_probabilities(
            self.n_nodes, exchangeable=False, wholeMatrix=False
        )
        if p_0.shape != p_hat.shape:
            raise ValueError(
                "The probability matrices have different shapes: {} and {}".format(
                    p_0.shape, p_hat.shape
                )
            )
        return np.mean((p_0 - p_hat) ** 2)

    def __str__(self):
        return "MSE on probabily matrix"


class MaeProbaEdge(ValueMetric):
    def __init__(self, n_nodes: int = 100) -> None:
        super().__init__()
        self.n_nodes = n_nodes

    def _compute(self, graphon: Graphon, graphon_2: Graphon, **kwargs) -> float:
        r"""Compute the MAE between the probability matrix of 2 different graphons.

        This follows the formula :math: `MAE = \frac{1}{n^2} \sum_{i,j} |p_{ij} - \hat{p}_{ij}|` where
        :math: `p_{ij}` is the probability of an edge between nodes :math: `i` and :math: `j` in the
        actual graphon and :math: `\hat{p}_{ij}` is the probability of an edge between nodes :math: `i`
        and :math: `j` in the graphon_2 graphon. Can be used to compare a theoretical graphon and an
        estimated graphon.

        Parameters
        ----------
        graphon : Graphon
            The theoretical graphon to compare to
        graphon_2 : Graphon
            The graphon_2 graphon

        Returns
        -------
        float
            The MAE between the probability matrix of the graphon and the graphon_2 graphon

        Raises
        ------
        ValueError
            If the probability matrices have different shapes
        """
        p_0 = graphon.get_edge_probabilities(self.n_nodes, exchangeable=False, wholeMatrix=False)
        p_hat = graphon_2.get_edge_probabilities(
            self.n_nodes, exchangeable=False, wholeMatrix=False
        )
        if p_0.shape != p_hat.shape:
            raise ValueError(
                "The probability matrices have different shapes: {} and {}".format(
                    p_0.shape, p_hat.shape
                )
            )
        return np.mean(np.abs(p_0 - p_hat))

    def __str__(self):
        return "MAE on probabily matrix"
