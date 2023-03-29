#########################################################
Graphon approximation with the political weblogs dataset
#########################################################


We will take a look at the political weblogs dataset :footcite:`adamic2005political` using the network histogram approximation 
from :cite:t:`olhede2014` and the SAS estimator from :cite:t:`chan2014`.

We consider the adjacency matrix representing the weblogs with at least one link to another weblog in the dataset. 

.. admonition:: Getting the data
   :class: toggle, hint

   This dataset is available from `the Netzschleuder network catalogue <https://networks.skewed.de/>`_  using 
   `this link <https://networks.skewed.de/net/polblogs/files/polblogs.csv.zip>`_. Here are two ways to get the data:

   .. tab:: Python

      Run the following code (you may need to first install the required packages with ``pip install pandas requests networkx zipfile``)

      .. code:: python

         import pandas as pd
         import networkx as nx
         import numpy as np
         import requests, io
         from zipfile import ZipFile

         url =  "https://networks.skewed.de/net/polblogs/files/polblogs.csv.zip"
         response = requests.get(url, stream=True)
         with ZipFile(io.BytesIO(response.content)) as myzip:
            with myzip.open(myzip.namelist()[0]) as myfile:
               df = pd.read_csv(myfile)

         df.columns = ["source", "target"]
         A = nx.to_numpy_array(nx.from_pandas_edgelist(df, "source", "target"))
         # we remove the diagonal to avoid self-loops
         A -= np.diag(np.diag(A))

   

   .. tab:: Download manually

      You can also download the data manually from the `Netzschleuder website <https://networks.skewed.de/>`_ , and load the 
      `edges.csv` file into an adjacency matrix using the `read_edgelist method from 
      networkx <https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.edgelist.read_edgelist.html>`_


      .. code:: console

         wget https://networks.skewed.de/net/polblogs/files/polblogs.csv.zip
         unzip polblogs.csv.zip

 

.. tab:: Network histogram estimator

   Following :cite:t:`olhede2014`, we use blocks of 72 nodes to bin the nodes. [2]_


   .. code:: python

      from pygraphon.estimators import NetworkHistogramEstimator
      estimator = NetworkHistogramEstimator(bandwithHist = 1/72)
      estimator.fit(A)

.. tab:: SAS estimator

   We let the SAS estimator choose the bandwidth automatically.

   .. code:: python

      from pygraphon.estimators import SAS
      estimator = SAS()
      estimator.fit(A)

   

We can now plot the estimated graphon :math:`\hat{f}:[0,1]^2 \mapsto [0,1]` and the estimated matrix of edge probabilities 
:math:`\hat{P}_{ij} = \hat{f}(\hat{\xi}_i,\hat{\xi}_j)`.


We first prepare the plot and import the necessary packages:

.. code:: python

   import matplotlib.pyplot as plt

   # create figure 
   fig, ax = plt.subplots(1, 2, figsize=(10, 5))

   # adjust for colorbar
   fig.subplots_adjust(right=0.85)
   cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])

   # details
   colormap = "jet"
   ax[1].set_title("Edge connectivity")
   ax[0].set_title("Estimated Graphon")


We can then plot the graphon and the edge probabilities using the following code:


.. code:: python

   from pygraphon.plots import plot as plot_step_graphon

   graphon = estimator.get_graphon()
   edge_probabilities = estimator.get_edge_connectivity()

   plot_step_graphon(graphon, fig=fig, ax=ax[0], colormap=colormap)
   im = ax[1].imshow(edge_probabilities, cmap=colormap)

   plt.colorbar(im, cax=cbar_ax)
   plt.show()

   


.. tab:: Network histogram estimator

   .. image:: nethist_pollblog.png

.. tab:: SAS estimator

   .. image:: sas_pollblog.png

.. footbibliography::


.. [2] In the paper introducing the method, the authors report a normalized log-likelihood of :math:`-2.8728` for this dataset. Ours is 
   slightly better, probably due to the randomness of the optimization procedure.