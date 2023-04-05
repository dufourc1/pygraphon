import io
import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests

from pygraphon.estimators import SAS
from pygraphon.plots import plot as plot_step_graphon

this_dir = os.path.dirname(os.path.abspath(__file__))

url =  "https://networks.skewed.de/net/polblogs/files/polblogs.csv.zip"
response = requests.get(url, stream=True)
with ZipFile(io.BytesIO(response.content)) as myzip:
   with myzip.open(myzip.namelist()[0]) as myfile:
      df = pd.read_csv(myfile)

df.columns = ["source", "target"]
A = nx.to_numpy_array(nx.from_pandas_edgelist(df, "source", "target"))
# we remove the diagonal to avoid self-loops
A -= np.diag(np.diag(A))

estimator = SAS()
estimator.fit(A)

graphon = estimator.get_graphon()
edge_probabilities = estimator.get_edge_connectivity()

# prepare plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])
colormap = "jet"

plot_step_graphon(graphon, fig=fig, ax=ax[0], colormap=colormap)
im = ax[1].imshow(edge_probabilities, cmap=colormap)
ax[1].set_title("Edge connectivity")
ax[0].set_title("Estimated graphon")
plt.colorbar(im, cax=cbar_ax)
plt.savefig(os.path.join(this_dir,"../docs/source/examples","sas_pollblog.png"))
plt.close()
