######
 Norm
######

=================================================================================
Difference Between Value Estimation and Function Estimation in Graphon Estimation
=================================================================================

Introduction
------------
When estimating a graphon, which is a mathematical object used to model random graphs, there are two main approaches: value estimation and function estimation. These approaches differ in terms of the specific aspects of the graphon they aim to estimate and the methods they employ. Let's explore each one in detail.

Value Estimation
----------------
Value estimation focuses on estimating the values of the graphon at individual points or pairs of points. A graphon is essentially a function defined on the unit square [0, 1] × [0, 1], representing the probability of an edge existing between any two points in a random graph. In value estimation, the primary goal is to estimate the values of the graphon at specific points or pairs of points, rather than estimating the entire graphon function itself.

To perform value estimation, one typically collects a set of graph samples or observations and analyzes the presence or absence of edges between the corresponding points. Statistical methods, such as maximum likelihood estimation or Bayesian inference, can then be employed to estimate the probabilities or intensities associated with the observed edges. These estimates provide insights into the values of the graphon at specific locations or pairs of locations.

The Mean Square Error (MSE) can be used as a measure of the accuracy of value estimation. Let's denote the estimated values of the graphon as :math:`\hat{W}(x,y)` and the true values as :math:`W(x,y)`, where :math:`x` and :math:`y` are points in the unit square. The MSE is given by the formula:

.. math::
    \text{{MSE}} = \frac{1}{{n^2}} \sum_{{i=1}}^n \sum_{{j=1}}^n (\hat{W}(x_i,y_j) - W(x_i,y_j))^2,

where :math:`n` is the number of nodes.

Function Estimation
-------------------
Function estimation aims to estimate the entire graphon function as a whole. Instead of focusing on individual values, the objective is to recover the underlying structure of the graphon across the entire unit square. This approach requires estimating a continuous function defined over the entire domain [0, 1] × [0, 1].

Function estimation techniques often involve nonparametric methods, such as kernel smoothing or spline interpolation. These methods leverage the observed graph samples to estimate the graphon's functional form and smoothness. The estimated graphon can then be used for various analyses, including simulating random graphs or calculating graph properties.

To evaluate the accuracy of function estimation, the Mean Integrated Squared Error (MISE) is commonly used. Let :math:`\hat{W}(x,y)` be the estimated graphon function and :math:`W(x,y)` be the true graphon function. The MISE is calculated as:

.. math::
    \text{{MISE}} = \frac{1}{{n^2}} \int_0^1 \int_0^1 (\hat{W}(x,y) - W(x,y))^2 \, dx \, dy,

where :math:`n` is the number of nodes or points.

Conclusion
----------
In summary, value estimation in graphon estimation aims to estimate specific values of the graphon at given points or pairs of points, while function estimation seeks to

Side Note: Ongoing Debate
-------------------------
It's important to note that the choice between value estimation and function estimation is not always clear-cut. There is an ongoing debate in the graphon estimation community about which approach is better. Some researchers argue that value estimation is more appropriate because it is more closely aligned with the original definition of a graphon as a function defined on the unit square. Others argue that function estimation is more useful because it allows for more flexibility in the choice of estimation methods and provides more insights into the underlying structure of the graphon. 

For further exploration of this topic, you may refer to the paper "An iterative step-function estimator for graphons" by Diana Cai. The paper can be accessed at `this link <https://arxiv.org/abs/1412.2129>`_.


API
---

.. currentmodule:: pygraphon.norm

.. automodule::  pygraphon.norm
    :private-members:
    :members: