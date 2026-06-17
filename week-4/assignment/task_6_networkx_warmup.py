import networkx as nx
import numpy as np

G = nx.karate_club_graph()

adj_matrix = nx.to_numpy_array(G)

edge_list = list(G.edges())

print(adj_matrix.shape)
print(edge_list[:5])