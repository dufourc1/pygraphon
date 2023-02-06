from pygraphon.generators import DummyGenerator


def test_dummy_generator():
    """Should generate a list of adjacency matrices."""
    generator = DummyGenerator()
    graphs = generator.generate(10, 5)
    assert len(graphs) == 5
    assert graphs[0].shape == (10, 10)