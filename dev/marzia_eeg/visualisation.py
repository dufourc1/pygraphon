import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, List

def plot_eeg_network(
    matrix: np.ndarray,
    labels: list[str],
    coords: list[Tuple],
    threshold: float = 50,
    folder_path: str = None,
    survival: bool = False):
    print(folder_path)
    G = nx.Graph()
    for i, label in enumerate(labels):
        G.add_node(label, pos=coords[i])

    tau = np.percentile(matrix, threshold)

    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i, j] > tau:
                G.add_edge(labels[i], labels[j], weight=matrix[i, j])

    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=700, with_labels=True, node_color='skyblue', edge_color=weights, width=1, edge_cmap=plt.cm.Blues)
    plt.title('EEG Connectivity Network')
    if folder_path:
        if survival:
            plt.savefig(folder_path + str(threshold) + 'Surv%'  + 'eeg_network.png')
        else:
            plt.savefig(folder_path + str(threshold)+ 'NSurv%' + 'eeg_network.png')
    plt.show()


def plot_dual_eeg_network(
    matrix1: np.ndarray, 
    matrix2: np.ndarray, 
    labels: List[str], 
    coords: List[Tuple], 
    threshold: float = 50, 
    folder_path: str = None,
    ):
    def create_network(matrix):
        G = nx.Graph()
        for i, label in enumerate(labels):
            G.add_node(label, pos=coords[i])
        tau = np.percentile(matrix, threshold)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i, j] > tau:
                    G.add_edge(labels[i], labels[j], weight=matrix[i, j])
        return G

    # Create networks for both matrices
    G1 = create_network(matrix1)
    G2 = create_network(matrix2)

    pos = nx.get_node_attributes(G1, 'pos')  # Assumes both graphs use the same positions

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Setup figure and axes
    
    # Plot for matrix1
    weights1 = [G1[u][v]['weight'] for u, v in G1.edges()]
    nx.draw(G1, pos, ax=axes[0], node_size=700, with_labels=True, node_color='skyblue',
            edge_color=weights1, width=1, edge_cmap=plt.cm.Blues)
    axes[0].set_title('EEG ConNet Surv')
    
    # Plot for matrix2
    weights2 = [G2[u][v]['weight'] for u, v in G2.edges()]
    nx.draw(G2, pos, ax=axes[1], node_size=700, with_labels=True, node_color='skyblue',
            edge_color=weights2, width=1, edge_cmap=plt.cm.Blues)
    axes[1].set_title('EEG ConNet NSurv')

    # Saving the plots if folder_path is specified
    if folder_path:
        plt.savefig(folder_path + str(threshold) + 'tau_comp_eeg_network.png')

    plt.show()

