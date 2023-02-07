"""Core functionnalities."""

from .common_graphon_functions import *  # noqa
from .Graphon import Graphon
from .StepGraphon import StepGraphon

common_graphon_names = ["graphon_" + name for name in common_graphons.keys()]
print(common_graphon_names)
classes = ["Graphon", "StepGraphon"]
__all__ = common_graphon_names + classes