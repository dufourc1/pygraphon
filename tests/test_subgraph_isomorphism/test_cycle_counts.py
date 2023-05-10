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
    if normalization == "s":
        theoretical_density = number_cycles / comb(n_nodes, length)
    if normalization == "t":
        theoretical_density = number_cycles * length * 2 / (n_nodes**length)
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


def test_cycle_counts_against_nx():
    """Test cycle counts against networkx cycle counts."""
    G = nx.erdos_renyi_graph(40, 0.3)
    A = nx.to_numpy_array(G)
    cycles_nx = nx.simple_cycles(G, 5)
    n_cycles = np.zeros(6)
    for c in cycles_nx:
        n_cycles[len(c)] += 1
    cc = CycleCount(5, normalization="s")
    _ = cc(A)
    assert np.array_equal(n_cycles[3:], cc.counts)


def test_erdos_renyi_densities():
    """Test cycle densities against theoretical values for ER."""
    p = 0.4
    A = nx.to_numpy_array(nx.erdos_renyi_graph(100, p, seed=13))
    cc = CycleCount(5, normalization="t")
    normalized = cc(A)
    theoretical = p ** np.arange(3, cc.L + 1)
    assert np.allclose(normalized, theoretical, atol=1e-3, rtol=1e-2)
