# Summer of Code: Graph Neural Networks & Recommendation Systems

**Name:** Lakshya Agrawal

**Roll No:** 25b2186

Welcome to my repository for the Summer of Code program! This repository contains all my code, mathematical proofs, and theoretical summaries documenting my journey from the foundational calculus of Deep Learning to advanced Graph Machine Learning and Recommendation Systems.

## Weekly Breakdown

### Week 1 & 2: Mathematical Foundations & MLPs from Scratch

The first two weeks focused on rigorously understanding the exact mathematical mechanics behind neural networks without relying on high-level libraries. This involved calculating complex gradients, Jacobians, and Hessians by hand to understand loss landscapes, and implementing a complete, fully functioning Multi-Layer Perceptron (MLP) using pure NumPy from scratch.

### Week 3: Transitioning to PyTorch & CNNs

Moving away from pure NumPy, this week introduced modern deep learning frameworks and spatial feature extraction. I learned PyTorch tensor manipulation, autograd mechanics, and module construction, and then applied these tools to design, train, and evaluate Convolutional Neural Networks (CNNs) on the MNIST image dataset.

### Week 4: Graph Machine Learning Foundations

This week marked a major paradigm shift from Euclidean data (like grids and images) to complex, relational graph topologies. I explored how directed, heterogeneous multi-graphs store real-world facts as Knowledge Graphs, built custom object-oriented graph data structures using NetworkX and standard Python, and proved that traditional spatial data can be modeled as a graph by converting the MNIST image dataset into a PyTorch Geometric (PyG) graph format.

### Week 5: Spectral Graph Theory & Deep GCNs

Diving into the rigorous mathematics of Graph Convolutional Networks, this week focused on message passing and analyzing architectural bottlenecks. I completed hand-calculated mathematical proofs involving Adjacency Matrices and the Normalized Graph Laplacian, built a manual GCN layer using only NumPy matrix multiplications, and used PyTorch Geometric (PyG) to train GCNs of varying depths on the Cora dataset to empirically demonstrate the phenomenon of over-smoothing.

