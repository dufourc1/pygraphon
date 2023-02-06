import networkx as nx

from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


class BaseGenerator:
    def generate(self, number_nodes: int, number_graphs: int):
        raise NotImplementedError()


class DummyGenerator(BaseGenerator):
    def generate(self, number_nodes: int, number_graphs: int):
        return [self._er(number_nodes) for _ in range(number_graphs)]

    def _er(self, number_nodes: int):
        return get_adjacency_matrix_from_graph(nx.erdos_renyi_graph(number_nodes, 0.5))
