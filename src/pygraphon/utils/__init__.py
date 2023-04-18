"""utils to work with graphon."""

from .utils_func import copy_func
from .utils_graph import (
    check_simple_adjacency_matrix,
    edge_density,
    get_adjacency_matrix_from_graph,
)
from .utils_graphon import compute_areas_histogram
from .utils_maths import EPS, bic, generate_all_permutations, log_likelihood
from .utils_matrix import (
    bound_away_from_one_and_zero_arrays,
    check_symmetric,
    permute_matrix,
    upper_triangle_values,
)

__all__ = [
    "copy_func",
    "generate_all_permutations",
    "check_symmetric",
    "permute_matrix",
    "upper_triangle_values",
    "bound_away_from_one_and_zero_arrays",
    "get_adjacency_matrix_from_graph",
    "check_simple_adjacency_matrix",
    "edge_density",
    "compute_areas_histogram",
    "bic",
    "log_likelihood",
]
