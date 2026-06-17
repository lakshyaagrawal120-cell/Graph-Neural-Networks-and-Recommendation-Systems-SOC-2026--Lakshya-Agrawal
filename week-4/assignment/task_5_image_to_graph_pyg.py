import torch
from torchvision import datasets
from torch_geometric.data import Data
import numpy as np

dataset = datasets.MNIST(root='./data', train=True, download=True)
img = dataset[0][0]
img_arr = np.array(img) / 255.0

h, w = img_arr.shape
node_features = torch.tensor(img_arr.flatten(), dtype=torch.float32).view(-1, 1)

edges = []
for i in range(h):
    for j in range(w):
        node_idx = i * w + j
        if i > 0:
            edges.append([node_idx, (i - 1) * w + j])
        if i < h - 1:
            edges.append([node_idx, (i + 1) * w + j])
        if j > 0:
            edges.append([node_idx, i * w + (j - 1)])
        if j < w - 1:
            edges.append([node_idx, i * w + (j + 1)])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
y = torch.tensor([dataset[0][1]], dtype=torch.long)

graph_data = Data(x=node_features, edge_index=edge_index, y=y)
print(graph_data)
print(graph_data.num_nodes)
print(graph_data.num_edges)