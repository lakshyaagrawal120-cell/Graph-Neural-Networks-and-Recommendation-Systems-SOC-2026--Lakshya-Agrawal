# Week 4 Assignment Summary: Graph Machine Learning Foundations

## Task 2: Demystifying Knowledge Graphs (KGs)
Knowledge Graphs (KGs) are structured representations of facts and interconnected descriptions of entities. From a graph machine learning perspective, they are directed, heterogeneous multi-graphs. Unlike homogeneous graphs where all nodes are identical in type, KGs contain various types of entities (nodes) and various types of relationships (edges). Information in KGs is stored in triplets consisting of a `(head, relation, tail)` structure—for example, `(DaVinci, Painted, MonaLisa)`. They are primarily utilized to embed real-world, semantic knowledge into machine learning systems for downstream tasks like link prediction, recommendation systems, or question answering.

## Task 3: The GNN Killers: Over-smoothing vs. Over-squashing

**Over-smoothing**
* **What it is:** A phenomenon where, as more layers are added to a Graph Neural Network (GNN), the node embeddings become increasingly indistinguishable from one another.
* **Why it happens:** Message passing operations mathematically resemble a low-pass filter or a random walk. With each successive layer, a node aggregates information from an exponentially growing structural neighborhood. Eventually, the aggregations mix so thoroughly that all node representations converge to the exact same average vector.
* **The Tension:** There is an inherent architectural tension in GNN design. Stacking more layers is theoretically necessary to expand the receptive field (allowing a node to gather context from distant parts of the graph). However, expanding this field actively destroys the node distinctiveness that is strictly required for accurate node-level classification.

**Over-squashing**
* **What it is:** A structural bottleneck issue where information from a massive computational tree fails to reach the target node effectively.
* **Why it happens:** In graphs with small-world properties or heavily connected hub nodes, the number of neighbors grows exponentially with each hop. When a GNN attempts to compress this massive volume of distant neighborhood features into a single, fixed-size hidden vector during the message passing phase, critical structural information gets "squashed" and lost.
* **How to solve them:** Both phenomena can be mitigated through specific architectural interventions. Solutions include introducing skip connections (to preserve original node features across deep layers), applying layer normalization, utilizing careful neighborhood sampling strategies (such as GraphSAGE), or implementing graph rewiring (modifying the underlying adjacency matrix before training to alleviate structural bottlenecks).

## Task 4: The Colab
The PyTorch Geometric Node Classification Colab was successfully executed and reviewed. The notebook demonstrated the standard PyG workflow, transitioning from basic dataset loading (Cora) to implementing, training, and evaluating Graph Convolutional Network (GCN) architectures for node-level predictions.

## Task 5: Paradigm Shift - Anything is a Graph
To demonstrate this paradigm, the MNIST dataset (previously used for tabular/image ML tasks) was converted into a PyTorch Geometric dataset. 
* **The Process:** Each 28x28 image was treated as a grid graph. Every pixel was initialized as a node with a feature vector representing its grayscale intensity. Edges were drawn strictly between physically adjacent pixels (up, down, left, right).
* **The Reflection:** Processing an image as a graph is highly logical when moving beyond rigid grid constraints. While Convolutional Neural Networks (CNNs) rely on perfect, uniform rectangular matrices, graph representations allow for irregular structures. By representing an image as a graph, it becomes possible to dynamically prune "empty" or black background pixels entirely. This creates a memory-efficient structure that focuses solely on the relevant strokes of the digits. Furthermore, this topological logic allows computer vision techniques to be applied to non-Euclidean surfaces (like 3D meshes or spheres) where traditional 2D convolutions fail.

## Task 7: Build Your Own Graph Class vs. PyG
A custom Python class was developed to handle basic graph construction, storing nodes and edges in standard dictionary structures. 
* **Compare & Contrast:** The custom class is highly intuitive and human-readable, tracking connections using standard object-oriented logic. However, PyTorch Geometric (`PyG`) handles graph data entirely differently under the hood to maximize computational efficiency. Instead of object-oriented dictionaries, PyG's `Data` object uses rigid, flattened tensors. Its most notable abstraction is the `edge_index`, which stores all connections in a Coordinate Format (COO) matrix of shape `[2, num_edges]`. While less intuitive for a human to read, this tensor-based abstraction is absolutely mandatory for executing message passing operations through massive parallelization on GPU hardware.
