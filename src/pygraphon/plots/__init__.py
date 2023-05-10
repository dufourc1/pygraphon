"""module for plotting graphons and matrices."""

from .plot_matrices import spy
from .plots_graphon import plot_graphon_function, plot_probabilities
from .utils import make_0_1, make_pretty

__all__ = [
    "plot_graphon_function",
    "plot_probabilities",
    "spy",
    "make_0_1",
    "make_pretty",
]
