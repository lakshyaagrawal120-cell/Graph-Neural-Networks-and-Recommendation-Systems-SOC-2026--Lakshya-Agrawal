import torch

A = torch.tensor([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
], dtype=torch.float32)

X = torch.tensor([
    [1.0, -1.0],
    [2.0, 0.5],
    [0.0, 3.0],
    [1.5, 1.5]
], dtype=torch.float32)

D = torch.sum(A, dim=1)
D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

A_norm = D_inv_sqrt @ A @ D_inv_sqrt

W = torch.randn(2, 4)

X_new = A_norm @ X @ W
X_new = torch.relu(X_new)

print(X_new)