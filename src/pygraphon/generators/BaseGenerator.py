import networkx as nx

from pygraphon.graphons import Graphon
from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


class BaseGenerator:
    def generate(self, number_nodes: int, number_graphs: int):
        raise NotImplementedError()


class DummyGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.graphon = Graphon(function=lambda x, y: 0.5)

    def generate(self, number_nodes: int, number_graphs: int):
        return [self.graphon.draw(rho=1.0, n=number_nodes) for _ in range(number_graphs)]
