"""utils to work with graphon."""

from .utils_elongation import (
    create_diagonal_partition_symmetric,
    create_symmetric_diagonal_squares_partition,
    get_elongation_list,
    compute_average_elongation,
)
from .utils_func import copy_func
from .utils_graph import (
    check_simple_adjacency_matrix,
    edge_density,
    get_adjacency_matrix_from_graph,
)
from .utils_graphon import check_consistency_graphon_shape_with_bandwidth, compute_areas_histogram
from .utils_maths import (
    EPS,
    aic,
    bic,
    elbow_point,
    fpe,
    generate_all_permutations,
    hqic,
    log_likelihood,
    mallows_cp,
    mdl,
)
from .utils_matrix import (
    bound_away_from_one_and_zero_arrays,
    check_symmetric,
    permute_matrix,
    scatter_symmetric_matrix,
    upper_triangle_values,
)
from .utils_ssm import AssociativeFullyRandom, generate_hierarchical_theta

__all__ = [
    "create_diagonal_partition_symmetric",
    "create_symmetric_diagonal_squares_partition",
    "get_elongation_list",
    "compute_average_elongation",
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
    "aic",
    "elbow_point",
    "mallows_cp",
    "hqic",
    "fpe",
    "mdl",
    "log_likelihood",
    "check_consistency_graphon_shape_with_bandwidth",
    "scatter_symmetric_matrix",
    "generate_hierarchical_theta",
    "AssociativeFullyRandom",
]
