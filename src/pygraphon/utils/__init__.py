"""utils to work with graphon."""

from .utils_func import copy_func
from .utils_graph import (
    check_simple_adjacency_matrix,
    edge_density,
    get_adjacency_matrix_from_graph,
)
from .utils_graphon import check_consistency_graphon_shape_with_bandwidth, compute_areas_histogram
from .utils_jit import log_likelihood
from .utils_maths import (
    EPS,
    aic,
    bic,
    elbow_point,
    fpe,
    generate_all_permutations,
    hqic,
    mallows_cp,
)
from .utils_matrix import (
    bound_away_from_one_and_zero_arrays,
    check_symmetric,
    permute_matrix,
    scatter_symmetric_matrix,
    upper_triangle_values,
)
from ..graphons.ssm_generators import AssociativeFullyRandom, generate_hierarchical_theta

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
    "aic",
    "elbow_point",
    "mallows_cp",
    "hqic",
    "fpe",
    "log_likelihood",
    "check_consistency_graphon_shape_with_bandwidth",
    "scatter_symmetric_matrix",
    "generate_hierarchical_theta",
    "AssociativeFullyRandom",
]
