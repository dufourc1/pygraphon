##########
 Graphons
##########


We provide general classes for two different types of graphon. General
graphons and step graphons. 



.. currentmodule:: pygraphon.graphons

.. autosummary::
   :nosignatures:
   :toctree: pygraphon.graphons
   
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
   
   





.. autoclass:: pygraphon.graphons.Graphon
	:noindex:

  	.. automethod:: draw
		:noindex:


