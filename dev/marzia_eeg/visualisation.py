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
    cluster_label: List[str],  # Cluster labels corresponding to brain regions
    threshold: float = 50,
    frequency: str = None,
    folder_path: str = None,
    ):
    # Define a color map for the clusters
    unique_clusters = list(set(cluster_label))
    colors = plt.cm.get_cmap('Pastel2', len(unique_clusters))  # Get a colormap that can provide enough distinct colors
    cluster_to_color = {cluster: colors(i) for i, cluster in enumerate(unique_clusters)}

    def create_network(matrix):
        G = nx.Graph()
        for i, label in enumerate(labels):
            G.add_node(label, pos=coords[i], cluster=cluster_label[i])  # Include cluster information
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
    clusters = [G1.nodes[label]['cluster'] for label in labels]  # Extract cluster labels for colors
    node_colors = [cluster_to_color[cluster] for cluster in clusters]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Setup figure and axes

    # Plot for matrix1
    weights1 = [G1[u][v]['weight'] for u, v in G1.edges()]
    nx.draw(G1, pos, ax=axes[0], node_size=700, with_labels=True, node_color=node_colors,
            edge_color=weights1, width=1, edge_cmap=plt.cm.Blues)
    axes[0].set_title('Surviving EEG at frequency ' + frequency + ' and threshold ' + str(threshold) + '%' if frequency else 'Surviving EEG')

    # Plot for matrix2
    weights2 = [G2[u][v]['weight'] for u, v in G2.edges()]
    nx.draw(G2, pos, ax=axes[1], node_size=700, with_labels=True, node_color=node_colors,
            edge_color=weights2, width=1, edge_cmap=plt.cm.Blues)
    axes[1].set_title('Surviving EEG at frequency ' + frequency + ' and threshold ' + str(threshold) + '%' if frequency else 'Non-surviving EEG')

    # Create a legend for the colors
    for cluster, color in cluster_to_color.items():
        plt.scatter([], [], color=color, label=cluster)
    plt.legend(title="Region", title_fontsize='13', fontsize='11', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.suptitle('Connectivity Network')

    # Saving the plots if folder_path is specified
    if folder_path:
        plt.savefig(folder_path + str(threshold) + 'tau_comp_eeg_network.png')

    plt.show()


def calculate_boundaries(sorted_list):
    unique_categories = sorted(set(sorted_list), key=sorted_list.index)  # Preserves order
    boundaries = [sorted_list.index(cat) for cat in unique_categories]
    boundaries += [len(sorted_list)]  # Add the last boundary
    return boundaries[1:]  # Skip the first boundary (zero)


def calculate_boundaries_and_labels(sorted_list):
    unique_categories = sorted(set(sorted_list), key=sorted_list.index)  # Preserves order
    boundaries = [sorted_list.index(cat) for cat in unique_categories]
    boundaries += [len(sorted_list)]  # Add the last boundary
    # Calculate midpoints for labels
    midpoints = [(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)]
    return boundaries[1:], midpoints, unique_categories  # Skip the first boundary (zero)


def plot_with_boundaries(data, boundaries, ax, cmap='jet', vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    for boundary in boundaries:
        ax.axhline(y=boundary-0.5, color='red', linestyle='--')  # Horizontal line
        ax.axvline(x=boundary-0.5, color='red', linestyle='--')  # Vertical line
    return im


def plot_with_boundaries_and_labels(data, boundaries, ax, labels, midpoints, cmap='jet', vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    for boundary in boundaries:
        ax.axhline(y=boundary-0.5, color='red', linestyle='--')  # Horizontal line
        ax.axvline(x=boundary-0.5, color='red', linestyle='--')  # Vertical line
    # Adding text labels
    for label, midpoint in zip(labels, midpoints):
        ax.text(midpoint, -5, label, ha='center', va='bottom', color='red', fontsize=10, rotation=90)
        ax.text(-10, midpoint, label, ha='right', va='center', color='red', fontsize=10)
    return im