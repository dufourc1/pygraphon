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

We also provide a list of pre-implemented common graphons:

.. list-table:: Common graphons
   :widths: 25 25 
   :header-rows: 1

   * - name 
     - function
   * - :py:class:`graphon_log`
     - :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`
   * - :py:class:`graphon_polynomial`
     - :math:`f(u,v) = \frac{1}{4}\left[u^2+v^2+u^{1 / 2}+v^{1 / 2}\right]`
   





.. autoclass:: pygraphon.graphons.Graphon.Graphon

   .. automethod:: draw


