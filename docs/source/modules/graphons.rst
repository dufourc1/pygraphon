##########
 Graphons
##########


We provide general classes for two different types of graphon. General
graphons and step graphons. 

.. currentmodule:: pygraphon.graphons

.. autosummary::
   :nosignatures:
   
   Graphon
   StepGraphon

We also provide a list of pre-implemented common graphons which are instances of :py:class:`Graphon`

.. list-table:: 
   :widths: 25 50

   * - :py:class:`graphon_product`
     - :math:`f(x,y) = x\cdot y`
   * - :py:class:`graphon_exp_07`
     - :math:`f(x,y) = \exp(x^{0.7}+y^{0.7})`
   * - :py:class:`graphon_polynomial`
     - :math:`f(x,y) = \frac{1}{4}\left[x^2+y^2+x^{1 / 2}+y^{1 / 2}\right]`
   * - :py:class:`graphon_mean`
     - :math:`f(x,y) = 0.5(x+y)` 
   * - :py:class:`graphon_logit_sum`
     - :math:`f(x,y) =\left[1+ e^{-10(x^2 + y^2)}\right]^{-1}`
   * - :py:class:`graphon_latent_distance`
     - :math:`f(x,y) = |x-y|`
   * - :py:class:`graphon_logit_max_power`
     - :math:`f(x,y) = \left[1+ e^{-\max(x,y)^2 + \min(x,y)^4}\right]^{-1}`
   * - :py:class:`exp_max_power`
     - :math:`f(x,y) = \exp(-\max(x,y)^{\frac{3}{4}})`
   * - :py:class:`graphon_exp_polynomial`
     - :math:`f(x,y) = \exp\left(-0.5\left[\min(x,y) + \sqrt{x} + \sqrt{y} \right]\right)`
   * - :py:class:`graphon_log_1_p`
     - :math:`f(x,y) = \log(1 + 0.5*\max(x,y))`
   
.. plot::
    :include-source: False
    :show-source-link: False
    :context: reset

    from pygraphon.graphons import common_graphons
    from pygraphon.plots import plot_probabilities, make_0_1
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for i, graphon_name in enumerate(common_graphons):
        plot_probabilities(common_graphons[graphon_name], ax=ax[i//5, i%5], show_colorbar=True)
        ax[i//5, i%5].set_title(graphon_name)
    plt.tight_layout()

.. automodule:: pygraphon.graphons
	:members:



Internal representation details:
--------------------------------

For statistical identifiability reasons, when given a function :math:`f:[0,1]^2 \mapsto [0,1]` we internaly use the scaled graphon 
:math:`\tilde{f}:[0,1]^2 \mapsto \mathbb{R}^{+}` defined as:

.. math::

   \tilde{f}(x,y) = \frac{f(x,y)}{\rho},


where :math:`\rho = \iint_{[0,1]^2} f(x,y)dxdy`.

In the :py:class:`Graphon` class, :py:attr:`~Graphon.graphon` is the scaled graphon :math:`\tilde{f}` and 
:py:attr:`~Graphon.initial_rho` is the value of :math:`\rho`.

