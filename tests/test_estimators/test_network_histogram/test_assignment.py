import numpy as np

from pygraphon.estimators.networkhistogram.greedy_readable import (
    EPS,
    Assignment,
    bernlikelihood,
    compute_edge_between_groups,
    update_realized_number,
)


def get_adj_and_labels():
    A = np.array(
        [
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
        ]
    )
    node_labels = np.array([1, 1, 1, 1, 2, 2, 2, 2]) - 1  # 0-indexed
    return A, node_labels


def test_bernlikelihood():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    for input in x:
        assert bernlikelihood(input) == input * np.log(input) + (1 - input) * np.log(1 - input)
    assert bernlikelihood(0) == bernlikelihood(EPS)
    assert bernlikelihood(1) == bernlikelihood(1 - EPS)


def test_compute_edge_between_groups():
    A, node_labels = get_adj_and_labels()
    indices_1 = np.where(np.array(node_labels) == 0)[0]
    indices_2 = np.where(np.array(node_labels) == 1)[0]
    assert compute_edge_between_groups(A, indices_1, indices_2) == 2
    assert compute_edge_between_groups(A, indices_2, indices_1) == 2
    assert compute_edge_between_groups(A, indices_1, indices_1) == 5 * 2
    assert compute_edge_between_groups(A, indices_2, indices_2) == 5 * 2


def test_update_counts():
    A, node_labels = get_adj_and_labels()
    realized_edges = np.array([[5, 2], [2, 5]])
    realized_edges = update_realized_number(realized_edges, node_labels, A, (1, 4))

    assert realized_edges[0, 0] == 2
    assert realized_edges[0, 1] == 8
    assert realized_edges[1, 0] == 8
    assert realized_edges[1, 1] == 2


def test_assignement():
    A, node_labels = get_adj_and_labels()
    assignment = Assignment(node_labels, A)
    assert np.all(assignment.realized == np.array([[5, 2], [2, 5]]))
    assert np.all(assignment.counts == np.array([[6, 16], [16, 6]]))
    assert np.all(assignment.theta == np.array([[5 / 6, 2 / 16], [2 / 16, 5 / 6]]))
    theoretical_ll = (
        6 * bernlikelihood(5 / 6) + 16 * bernlikelihood(2 / 16) + 6 * bernlikelihood(5 / 6)
    )
    assert assignment.likelihood == theoretical_ll


def test_update_assignment():
    A, node_labels = get_adj_and_labels()
    assignment = Assignment(node_labels, A)
    update = (1, 4)
    theoretical_ll_post_update = (
        2 * (2 * np.log(2 / 6) + np.log(4 / 6) * 4) + 8 * np.log(8 / 16) * 2
    )
    new_assignment = Assignment(np.array([1, 2, 1, 1, 1, 2, 2, 2]), A)
    print(assignment.realized)
    assignment.update(update, A)

    print(assignment.realized)
    print(new_assignment.realized)

    assert np.all(assignment.labels == new_assignment.labels)
    assert np.all(assignment.realized == new_assignment.realized)
    assert np.all(new_assignment.realized == np.array([[2, 8], [8, 2]]))
    assert np.all(assignment.counts == new_assignment.counts)
    assert np.all(new_assignment.counts == np.array([[6, 16], [16, 6]]))
    assert np.all(assignment.theta == new_assignment.theta)
    assert np.all(new_assignment.theta == np.array([[2 / 6, 8 / 16], [8 / 16, 2 / 6]]))
    assert assignment.likelihood == new_assignment.likelihood == theoretical_ll_post_update
