# Week 7 Project Summary: Robust GraphSAGE Recommendation System

**Project:** Graph Neural Networks and Recommendation Systems (SOC 2026)
**Component:** Full-Stack Bipartite GNN Recommendation Pipeline
**Dataset:** MovieLens 100k

---

## 🎯 Executive Summary
For the pre-finale week, the objective was to upgrade a placeholder GNN architecture into a robust, full-stack recommendation engine. This was successfully achieved by enhancing a PyTorch-based Bipartite GraphSAGE model, bridging it to a FastAPI backend, and visualizing real-time node embedding similarities via a React/Vite web interface.

## 🛠️ Architectural & Pipeline Enhancements
* **Train/Test Data Split:** Overhauled the training loop to implement a strict 80/20 train/test split to benchmark rating prediction deviations accurately on unseen data.
* **Dropout Regularization:** Injected a `nn.Dropout(0.2)` layer into the Multilayer Perceptron (MLP) prediction head to improve generalization capabilities and prevent memorization.
* **Evaluation Metrics:** Implemented real-time Root Mean Square Error (RMSE) calculations during evaluation to quantify predictive accuracy.
* **Dynamic Anomaly Detection:** Engineered an outlier detection algorithm in the backend that calculates rating velocity, variance, and genre contradictions to flag anomalous users.

---

## 📊 Hyperparameter Tuning & Observations

The model was tested across different configurations to observe convergence and accuracy behaviors on the MovieLens dataset.

| Configuration Test | Epochs | Train Loss (MSE) | Test RMSE | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Standard)** | 10 | 0.8576 | 0.9263 | The standard configuration established a solid baseline, proving the model rapidly achieves sub-1.0 error margins. |
| **Baseline (Extended)** | 25 | 0.8116 | 0.9230 | Extending epochs allowed the loss to drop further, showing strong convergence without immediate overfitting. |
| **Embedding Dim = 32** | 10 | 0.8465 | 0.9228 | Halving the embedding dimension yielded the best overall Test RMSE, suggesting the model successfully avoids overfitting with a slightly smaller parameter capacity. |
| **Learning Rate = 0.005** | 10 | 0.8440 | 0.9363 | A lower learning rate resulted in the highest RMSE, indicating the model requires more epochs to take enough gradient steps to reach optimal convergence. |
| **Batch Size = 1024** | 10 | 0.8574 | 0.9255 | Doubling the batch size maintained a highly competitive RMSE while processing more data simultaneously, though slightly less accurate than the dimension test. |

---

## 🚀 Full-Stack Integration Results
The final trained `.pth` embeddings successfully power the React frontend. The system processes explicit UI preferences, normalizes embedding profiles, and utilizes cosine similarity to generate dynamic, real-time movie recommendations. Additionally, the admin dashboard effectively visualizes the backend outlier logic, allowing for direct database manipulation to keep the GNN training loop healthy.
