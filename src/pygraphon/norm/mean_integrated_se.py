from ot.gromov import gromov_wasserstein2

from pygraphon.graphons import Graphon

from .BaseMetric import ValueMetric


class MISE(ValueMetric):

    def _compute(self, graphon: Graphon, graphon_2: Graphon, n=1000, **kwargs) -> float:
        theta = graphon.get_edge_probabilities(n, False)
        theta_2 = graphon_2.get_edge_probabilities(n, False)
        return gromov_wasserstein2(theta, theta_2)
