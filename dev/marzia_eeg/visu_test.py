import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple
from data_extract import get_band
from visualisation import plot_with_boundaries, plot_with_boundaries_and_labels, calculate_boundaries, calculate_boundaries_and_labels
from electrodes_data import sorted_by_region_then_hemi, sorted_by_hemisphere, region_indices, hemisphere_indices


# Assuming region_indices and hemisphere_indices are lists of electrode indices\n",
region_boundaries = calculate_boundaries([region for _, region, _ in sorted_by_region_then_hemi])
hemisphere_boundaries = calculate_boundaries([hemi for _, _, hemi in sorted_by_hemisphere])
# Assuming region_indices and hemisphere_indices are lists of electrode indices\n",
region_boundaries, region_midpoints, region_labels = calculate_boundaries_and_labels([region for _, region, _ in sorted_by_region_then_hemi])
hemisphere_boundaries, hemisphere_midpoints, hemisphere_labels = calculate_boundaries_and_labels([hemi for _, _, hemi in sorted_by_hemisphere])

plt.rcParams.update({
    'figure.figsize': (10, 10),
    'savefig.dpi': 300,
    'font.size': 20,
    'font.family': 'Arial',
    'figure.autolayout': True
})

def plot_combined_visualization(
    graphon_surv: np.ndarray,
    graphon_nsurv: np.ndarray,
    labels: List[str], 
    coords: List[Tuple], 
    cluster_label: List[str],
    permutation: str = None,
    threshold: float = 50,
    frequency: str = None,
    folder_path: str = None,
    cmap: str = 'binary',
    title_plot: str = None
):
    # Setup the figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    min_val = np.nanmin(np.concatenate([graphon_surv, graphon_nsurv]))
    max_val = np.nanmax(np.concatenate([graphon_surv, graphon_nsurv])) 

    # plot graphons
    for idx, (graphon, title) in enumerate(zip([graphon_surv, graphon_nsurv], ['Survivors', 'Non-survivors'])):
        if idx == 0:
            if permutation == 'regions':
                im = plot_with_boundaries_and_labels(graphon[region_indices, :][:, region_indices],
                                                    boundaries=region_boundaries,
                                                    midpoints=region_midpoints,
                                                    labels=region_labels,
                                                    cmap=cmap,
                                                    ax=axs[idx, 0],
                                                    vmin=min_val,
                                                    vmax=max_val)
            elif permutation == 'hemispheres':
                im = plot_with_boundaries_and_labels(graphon[hemisphere_indices, :][:, hemisphere_indices],
                                                    boundaries=hemisphere_boundaries,
                                                    midpoints=hemisphere_midpoints,
                                                    labels=hemisphere_labels,
                                                    cmap=cmap,
                                                    ax=axs[idx, 0],
                                                    vmin=min_val,
                                                    vmax=max_val)
        else:
            if permutation == 'regions':
                im = plot_with_boundaries(graphon[region_indices, :][:, region_indices],
                                        boundaries=region_boundaries,
                                        cmap=cmap,
                                        ax=axs[idx, 0],
                                        vmin=min_val,
                                        vmax=max_val)
            elif permutation == 'hemispheres':
                im = plot_with_boundaries(graphon[hemisphere_indices, :][:, hemisphere_indices],
                                        boundaries=hemisphere_boundaries,
                                        cmap=cmap,
                                        ax=axs[idx, 0],
                                        vmin=min_val,
                                        vmax=max_val)

        #axs[idx, 0].set_title(title)
        plt.colorbar(im, ax=axs[idx, 0], fraction=0.046, pad=0.04)

    # plot network
    unique_clusters = list(set(cluster_label))
    colors = plt.cm.get_cmap('Pastel2', len(unique_clusters))
    cluster_to_color = {cluster: colors(i) for i, cluster in enumerate(unique_clusters)}

    def create_network(matrix):
        G = nx.Graph()
        for i, label in enumerate(labels):
            G.add_node(label, pos=coords[i], cluster=cluster_label[i])
        tau = np.percentile(matrix, threshold)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i, j] > tau:
                    G.add_edge(labels[i], labels[j], weight=matrix[i, j])
        return G

    # Create networks
    G1 = create_network(graphon_surv)
    G2 = create_network(graphon_nsurv)
    pos = nx.get_node_attributes(G1, 'pos')
    clusters = [G1.nodes[label]['cluster'] for label in labels]
    node_colors = [cluster_to_color[cluster] for cluster in clusters]
    node_size = 700
    # Network plots
    for idx, (G, title) in enumerate(zip([G1, G2], ['Surviving EEG Network', 'Non-surviving EEG Network'])):
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw(G, pos, ax=axs[idx, 1], node_size=node_size, with_labels=True, node_color=node_colors,
                edge_color=weights, width=1, edge_cmap=plt.cm.Blues)
        #axs[idx, 1].set_title(title + ' at frequency ' + (frequency if frequency else "N/A") + ' and threshold ' + str(threshold) + '%')

    # Create a legend for the colors
    for cluster, color in cluster_to_color.items():
        plt.scatter([], [], color=color, label=cluster)
    plt.legend(title="Region", title_fontsize='30', fontsize='20', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.suptitle(title_plot if title_plot else 'Combined Visualization')
    plt.tight_layout()

    # Saving the plots if folder_path is specified
    if folder_path:
        plt.savefig(folder_path + 'combined_eeg_visualization.png')

    plt.show()
