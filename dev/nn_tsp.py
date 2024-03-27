import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

"""
scipy.spatial.distance.cdist function to calculate
the distance matrix between the points and the networkx
library to find the minimum spanning tree.
The Hamiltonian cycle is then obtained from the minimum
spanning tree by adding the first edge to the tree.
"""

def nn_TSP(points, n_points, n_features, init):
    dist_matrix = cdist(points, points, metric='euclidean')
    G = nx.from_numpy_array(dist_matrix)
    MST = nx.minimum_spanning_tree(G)
    edges = list(MST.edges)
    edges = np.array(edges)
    added_first = False
    path = [init]
    while True:
        if not added_first:
            added_first = True
            for edge in edges:
                if edge[0] == init:
                    path.append(edge[1])
                    edges = np.delete(edges, np.where(edges == edge), axis=0)
                    break
        else:
            new_edges = []
            for edge in edges:
                v = edge[0]
                w = edge[1]
                if w in path:
                    new_edges.append(edge)
            if len(new_edges) == 0:
                break
            else:
                edges = np.array(new_edges)
            for edge in new_edges:
                v = edge[0]
                w = edge[1]
                if v == path[-1]:
                    path.append(w)
                    edges = np.delete(edges, np.where(edges == edge), axis=0)
                    break
    path = np.array(path)
    path = path.reshape(n_points, 1)
    return path