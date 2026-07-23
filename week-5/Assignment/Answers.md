# 📚 Week 4 Answers: GCNs & Spectral Graph Theory

**Google Colab Link:** [https://colab.research.google.com/drive/1_jBsMpCZZsoz9ET7SxEiZmQXlwvUaIXR?usp=sharing]

## Q1. Normalizing the Adjacency Matrix

**1. Write the adjacency matrix** $A$ **and degree matrix** $D$
For the 3-node path graph (1 — 2 — 3):

$$
A = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad D = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

**2. Add self-loops to get** $\tilde{A}$ **and** $\tilde{D}$

$$
\tilde{A} = A + I_3 = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}
$$

$$
\tilde{D} = D + I_3 = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 2 \end{pmatrix}
$$

**3. Compute the symmetric normalized matrix** $\hat{A}$
First, we find the inverse square root of the degree matrix $\tilde{D}^{-1/2}$:

$$
\tilde{D}^{-1/2} = \begin{pmatrix} 1/\sqrt{2} & 0 & 0 \\ 0 & 1/\sqrt{3} & 0 \\ 0 & 0 & 1/\sqrt{2} \end{pmatrix}
$$

Now we multiply $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$:

$$
\hat{A} = \begin{pmatrix} 1/\sqrt{2} & 0 & 0 \\ 0 & 1/\sqrt{3} & 0 \\ 0 & 0 & 1/\sqrt{2} \end{pmatrix} \begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix} \begin{pmatrix} 1/\sqrt{2} & 0 & 0 \\ 0 & 1/\sqrt{3} & 0 \\ 0 & 0 & 1/\sqrt{2} \end{pmatrix}
$$

$$
\hat{A} = \begin{pmatrix} 1/2 & 1/\sqrt{6} & 0 \\ 1/\sqrt{6} & 1/3 & 1/\sqrt{6} \\ 0 & 1/\sqrt{6} & 1/2 \end{pmatrix}
$$

**4. Conceptual: High vs. Low Degree Weighting**
Because we divide the connection by $\sqrt{\tilde{d}_i \tilde{d}_j}$, the message passed from a high-degree neighbor is mathematically scaled down (weighted less) compared to a low-degree neighbor. This is a sensible design choice because highly connected "hub" nodes would otherwise dominate the feature representations of all their neighbors, washing out local, distinct signals and leading to a loss of useful graph topology.

## Q2. Implementing One GCN Layer

**2. Quick check: Compute** $(\hat{A}X)W$ **or** $\hat{A}(XW)$ **first?**
When $N \gg F$, you should generally compute $\hat{A}(XW)$ first.
*Why?* The operation $XW$ projects the node features into the new hidden dimension *before* broadcasting them across the graph. Multiplying the $N \times F$ matrix by the $F \times F_{out}$ weight matrix is a fast, dense operation resulting in an $N \times F_{out}$ matrix. Once transformed, you then perform the sparse neighborhood aggregation $\hat{A} \times (XW)$. If you computed $(\hat{A}X)$ first, you would be doing an expensive neighborhood aggregation on the larger original feature space, which is far less efficient.

## Q3. Why Do We Need Self-Loops?

**1. Short proof: Eigenvalues of** $D^{-1/2}AD^{-1/2}$ **lie in** $[-1, 1]$
We know that the normalized graph Laplacian is defined as $L = I_N - D^{-1/2}AD^{-1/2}$, and spectral graph theory tells us that the eigenvalues of $L$ always fall in the range $[0, 2]$.
Let $\lambda$ be an eigenvalue of $D^{-1/2}AD^{-1/2}$.
This means $1 - \lambda$ is the corresponding eigenvalue of $L$.

$$
0 \le 1 - \lambda \le 2
$$

Subtracting 1 from all sides:

$$
-1 \le -\lambda \le 1
$$

Multiplying by -1 (which flips the inequalities):

$$
1 \ge \lambda \ge -1
$$

Therefore, the eigenvalues of $D^{-1/2}AD^{-1/2}$ lie strictly in $[-1, 1]$.

**3. Conceptual: Missing the self-loop** $I_N$
Without $I_N$, the adjacency matrix has 0s on its diagonal. During the message-passing (matrix multiplication) phase, a node would aggregate the features of all its connected neighbors but would completely ignore its *own* previous features. The node's current state would be entirely wiped out and replaced by its neighborhood average, making it impossible to retain its own unique identity across layers.

## Q4. Over-Smoothing in Deep GCNs

**4. Conceptual: Why stacking layers hurts performance**
Message passing in a GCN is mathematically equivalent to taking a step in a random walk or applying a low-pass filter over the graph's structure. Each time a layer aggregates features, a node absorbs information from its neighbors. If you repeat this process too many times (stacking deep layers), the information mixes so thoroughly that every node effectively gathers the average of the entire connected component. When all node embeddings converge to the exact same average vector, they lose the distinctiveness required to be accurately classified into different categories. This is the essence of over-smoothing.
