"""Implementation of common graphon functions, as found in Table 1 of [1].

The common_graphons dictionary is ordered as in the paper.

[1]: Chan, Stanley, and Edoardo Airoldi. "A consistent histogram estimator for
    exchangeable graph models." International Conference on Machine Learning.
    PMLR, 2014.
"""
import numpy as np
from .Graphon import Graphon
from .StepGraphon import StepGraphon
from pygraphon.utils.utils_ssm import generate_hierarchical_theta, AssociativeFullyRandom#, graph_celtic, graph_concentric_circle, graph2_celtic 

# from Airoldi paper
graphon_product = Graphon(function=lambda x, y: x * y, repr="f(x,y) = x*y")
graphon_exp_07 = Graphon(
    function=lambda x, y: np.exp(-(x**0.7 + y**0.7)),
    repr="exp(-x^0.7 - y^0.7)",
)
graphon_polynomial = Graphon(
    function=lambda x, y: 0.25 * (x**2 + y**2 + np.sqrt(x) + np.sqrt(y)),
    repr="0.25(x^2 + y^2 + sqrt(x) + sqrt(y))",
)
graphon_mean = Graphon(function=lambda x, y: 0.5 * (x + y), repr="0.5(x + y)")
graphon_logit_sum = Graphon(
    function=lambda x, y: 1 / (1 + np.exp(-10 * (x**2 + y**2))),
    repr="1/(1 + exp(-10(x^2 + y^2)))",
)
graphon_latent_distance = Graphon(function=lambda x, y: np.abs(x - y), repr="|x - y|")
graphon_logit_max_power = Graphon(
    function=lambda x, y: 1 / (1 + np.exp(-(np.maximum(x, y) ** 2 + np.minimum(x, y) ** 4))),
    repr="1/(1 + exp(-(max(x,y)^2 + min(x,y)^4)))",
)
graphon_exp_max_power = Graphon(
    function=lambda x, y: np.exp(-np.maximum(x, y) ** (3 / 4)), repr="exp(-max(x,y)^(3/4))"
)
graphon_exp_polynomial = Graphon(
    function=lambda x, y: np.exp(-0.5 * (np.minimum(x, y) + np.sqrt(x) + np.sqrt(y))),
    repr="exp(-0.5(min(x,y) + sqrt(x) + sqrt(y)))",
)
graphon_log_1_p = Graphon(
    function=lambda x, y: np.log1p(0.5 * np.maximum(x, y)), repr="log(1 + 0.5*max(x,y))"
)

#One graphon to make usvt go brr
graphon_sinusoidal = Graphon(
    function=lambda x, y: 0.5+0.5*np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
    repr="sinusoidal",
)
#SBM
k = 15
m = 3
theta_min_intra = 0.5
theta_max_intra = 0.7
theta_min_meta_intra = 0.4
theta_max_meta_intra = 0.4
theta_min_inter = 0.1
theta_max_inter = 0.3

theta = AssociativeFullyRandom(
                    K=k,
                    min_inside_classes=theta_min_intra,
                    max_inside_classes=theta_max_intra,
                    min_outside_classes=theta_min_inter,
                    max_outside_classes=theta_max_inter,
                )
graphon_assortative_sbm = StepGraphon(theta.Theta, bandwidthHist=1 /k)
graphon_assortative_sbm.repr = "assortative_sbm"

theta = AssociativeFullyRandom(
                    K=k,
                    min_inside_classes=theta_min_inter,
                    max_inside_classes=theta_max_inter,
                    min_outside_classes=theta_min_intra,
                    max_outside_classes=theta_max_intra,
                )
graphon_dissassortative_sbm = StepGraphon(theta.Theta, bandwidthHist=1 /k)             
graphon_dissassortative_sbm.repr = "dissassortative_sbm"

theta = generate_hierarchical_theta(k, m, theta_min_intra, theta_max_intra, theta_min_meta_intra, theta_max_meta_intra, theta_min_inter, theta_max_inter)
graphon_hsbm = StepGraphon(theta, bandwidthHist=1 /k)
graphon_hsbm.repr = "hsbm"

theta = generate_hierarchical_theta(k, m, theta_min_intra, theta_max_intra, 0.25, 0.25, 0.15, 0.15)
graphon_hsbm_partition = StepGraphon(theta, bandwidthHist=1 /k)
graphon_hsbm_partition.repr = "hsbm_partition"

theta = generate_hierarchical_theta(15, 1, 0.7, 0.7, 0.2, 0.2, 0.2, 0.2)
graphon_partition = StepGraphon(theta, bandwidthHist=1 /15)
graphon_partition.repr = "partition"
#SSM
"""
graphon_celtic = Graphon(
    function=lambda x, y: graph_celtic(x, y, [0.25,0.75,0.1], [0.75, 0.5, 0.25, 0.1]),
    repr="celtic",
)
graphon_celtic2 = Graphon(
    function=lambda x, y: graph2_celtic(x, y, [0.25,0.75,0.2,0.25,0.25,0.2], [0.75, 0.5, 0.25, 0.1, 0.05]),
    repr="celtic2",
)
graphon_concentric_circle = Graphon(
    function=lambda x, y: graph_concentric_circle(
        x, y, [0.25, 0.75, 0.05, 0.1, 0.15, 0.2], [0.75, 0.5, 0.25, 0.1, 0.05]
    ),
    repr="concentric_circle",
)
"""

common_graphons = {
    "product": graphon_product,
    "exp_07": graphon_exp_07,
    "polynomial": graphon_polynomial,
    "mean": graphon_mean,
    "logit_sum": graphon_logit_sum,
    "latent_distance": graphon_latent_distance,
    "logit_max_power": graphon_logit_max_power,
    "exp_max_power": graphon_exp_max_power,
    "exp_polynomial": graphon_exp_polynomial,
    "log_1_p": graphon_log_1_p,
    "sinusoidal": graphon_sinusoidal,
    #"celtic": graphon_celtic,
    #"concentric_circle": graphon_concentric_circle,
    #"celtic2": graphon_celtic2,
    "assortative_sbm": graphon_assortative_sbm,
    "dissassortative_sbm": graphon_dissassortative_sbm,
    "hsbm": graphon_hsbm,
    "hsbm_partition": graphon_hsbm_partition,
    "partition": graphon_partition,
}
