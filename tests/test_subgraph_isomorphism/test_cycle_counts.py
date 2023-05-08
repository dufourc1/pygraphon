import networkx as nx
import numpy as np
import pytest
from scipy.special import comb

from pygraphon.subgraph_isomorphism import CycleCount


def make_cycle(length: int, n_nodes: int, n_cycles: int) -> np.ndarray:
    """Create a graph with n_nodes and n_cycles of length L."""
    G = nx.empty_graph(n_nodes)
    for i in range(n_cycles):
        nx.add_cycle(G, range(i * length, (i + 1) * length))
    return nx.to_numpy_array(G)


@pytest.mark.parametrize("length", [3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("normalization", ["s", "t"])
@pytest.mark.parametrize("number_cycles", [1, 2, 3, 4, 5])
def test_cycle_counts(length, normalization, number_cycles):
    """Test cycle counts for multiple cycle of length L."""
    n_nodes = 300
    A = make_cycle(length, n_nodes, number_cycles)
    theoretical_density = number_cycles / comb(n_nodes, length)
    if normalization == "t":
        theoretical_density = theoretical_density ** (1 / length)
    cc = CycleCount(length, normalization=normalization)
    normalized = cc(A)
    assert cc.counts[-1] == number_cycles
    assert normalized[-1] == pytest.approx(theoretical_density, abs=1e-7)


def test_cycle_counts_outside_boundary():
    """Test raise error for cycle counts outside boundary."""
    for L in [1, 2]:
        with pytest.raises(ValueError):
            CycleCount(L)

    with pytest.raises(NotImplementedError):
        CycleCount(10)
