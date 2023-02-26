import numpy as np

from pygraphon.estimators.networkhistogram.graphest_fastgreedy import (
    graphest_fastgreedy,
)
from pygraphon.estimators.networkhistogram.greedy_readable import greedy_opt
from pygraphon.estimators.networkhistogram.nethist import _first_guess_blocks
from pygraphon.graphons import graphon_logit_sum
from pygraphon.utils.utils_graph import edge_density

# Draw a graph from the graphon with a given density and number of vertices, see GraphonAbstract.py
A = graphon_logit_sum.draw(rho=1, n=400, exchangeable=False)

n = A.shape[0]
rho = edge_density(A)
n_obs = int(n * (n - 1) / 2)
starting_labels = _first_guess_blocks(A, 33, rho / 4)

_ = greedy_opt(A, starting_labels, maxNumRestarts=30)
