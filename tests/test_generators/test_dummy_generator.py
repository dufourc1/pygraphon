import numpy as np

from pygraphon.generators import DummyGenerator


def test_dummy_generator_shapes():
    """Should generate a list of adjacency matrices."""
    generator = DummyGenerator()
    graphs = generator.generate(10, 5)
    assert len(graphs) == 5
    assert graphs[0].shape == (10, 10)


def test_dummy_generator_edge_density():
    """Should generate a list of adjacency matrices with the correct edge density."""
    generator = DummyGenerator()
    graphs = generator.generate(100, 1)
    assert np.isclose(np.sum(graphs[0]) / (100 * 100), 0.5, atol=0.05)
