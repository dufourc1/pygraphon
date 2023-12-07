"""Implementation of common graphon functions, as found in Table 1 of [1].

The common_graphons dictionary is ordered as in the paper.

[1]: Chan, Stanley, and Edoardo Airoldi. "A consistent histogram estimator for
    exchangeable graph models." International Conference on Machine Learning.
    PMLR, 2014.
"""
import numpy as np

from .Graphon import Graphon

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

# One graphon to make usvt go brr
graphon_sinusoidal = Graphon(
    function=lambda x, y: 0.5 + 0.5 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),
    repr="sinusoidal",
)


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
}
