import numpy as np

from .Graphon import Graphon

graphon_product = Graphon(function=lambda x, y: x * y)
graphon_log = Graphon(function=lambda x, y: np.log1p(0.5 * max(x, y)))
