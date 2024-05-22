import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple

def plot_eeg_network(
    matrix: np.ndarray,
    labels: list[str],
    coords: list[Tuple],
    threshold: float = 0.5):
    # Create graph
    G = nx.Graph()
    for i, label in enumerate(labels):
        G.add_node(label, pos=coords[i])

    # Add edges based on the connectivity matrix threshold
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i, j] > threshold:
                G.add_edge(labels[i], labels[j], weight=matrix[i, j])

    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=700, with_labels=True, node_color='skyblue', edge_color=weights, width=1, edge_cmap=plt.cm.Blues)
    plt.title('EEG Connectivity Network')
    plt.show()