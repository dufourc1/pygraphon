import networkx as nx
import numpy as np

from pygraphon.estimators.networkhistogram.greedy_readable import (
    EPS,
    Assignment,
    bernlikelihood,
    compute_edge_between_groups,
    update_realized_number,
)
from pygraphon.utils.utils_graph import get_adjacency_matrix_from_graph


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
    new_assignment = Assignment(np.array([0, 1, 0, 0, 0, 1, 1, 1]), A)
    assignment.update(update, A)

    assert np.all(assignment.labels == new_assignment.labels)
    assert np.all(assignment.realized == new_assignment.realized)
    assert np.all(new_assignment.realized == np.array([[2, 8], [8, 2]]))
    assert np.all(assignment.counts == new_assignment.counts)
    assert np.all(new_assignment.counts == np.array([[6, 16], [16, 6]]))
    assert np.all(assignment.theta == new_assignment.theta)
    assert np.all(new_assignment.theta == np.array([[2 / 6, 8 / 16], [8 / 16, 2 / 6]]))
    assert assignment.likelihood == new_assignment.likelihood == theoretical_ll_post_update


def test_update_realized_numnber_vs_assignment_creation():
    n_per_group = 100
    seed = 17
    np.random.seed(seed)
    A = get_adjacency_matrix_from_graph(
        nx.stochastic_block_model(
            [n_per_group, n_per_group, n_per_group],
            [[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]],
            seed=seed,
        )
    )
    labels = np.array([0] * n_per_group + [1] * n_per_group + [2] * n_per_group)
    assignment = Assignment(labels, A)

    for i in range(n_per_group):
        realized_before = np.copy(assignment.realized)
        swap = np.random.choice(A.shape[0], 2, replace=False)
        new_labels = np.copy(assignment.labels)
        assignment.update((swap[0], swap[1]), A)
        new_labels[swap[0]], new_labels[swap[1]] = new_labels[swap[1]], new_labels[swap[0]]
        new_assignment = Assignment(new_labels, A)
        if not np.all(assignment.realized == new_assignment.realized):
            print("swap: ", swap)
            indices_wrong = np.where(assignment.realized != new_assignment.realized)[0]
            print("indices wrong: \n", indices_wrong)
            print("assignment: \n", assignment.realized)
            print("new assignment: \n", new_assignment.realized)
            print("realized before: \n", realized_before)
            print(A)
            print("labels: \n", assignment.labels)

        assert np.all(assignment.labels == new_assignment.labels)
        assert np.all(assignment.realized == new_assignment.realized)
        assert np.all(assignment.counts == new_assignment.counts)
        assert np.all(assignment.theta == new_assignment.theta)
        assert assignment.likelihood == new_assignment.likelihood
