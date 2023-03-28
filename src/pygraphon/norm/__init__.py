"""norm for graphons."""

from pygraphon.norm import BaseMetric
from pygraphon.norm.AUC_PR import AUCEdge, AUPRCEdge
from pygraphon.norm.MSE_MAE import MseProbaEdge, MaeProbaEdge
__all__ = classes = [
    "BaseMetric",
    "MseProbaEdge",
    "MaeProbaEdge",
    "AUCEdge",
    "AUPRCEdge",
]
