## About

The goal is to determine if a reinforcement learning controller is able to learn when it's beneficial to probe federated learning clients. The controller must choose to pass on the client, rank them by their (possibly stale) historical loss, or to run a fresh probing epoch. The aim is to reduce resource use without also reducing accuracy.

Currently Implemented:
- A PyTorch Fashion-MNIST simulator which has 20 non-IID clients (Dirichlet α  0.5)
  - Three actions per client (SKIP, HIST, PROBE)
  - Simulated compute/communication cost accounting
- Always-probe: a simplified FedMarl version
- Simplified UCB-CS: use past loss (history) as the only metric
- Adaptive controller: each client has a separate DQN agent, which is trained on a shared team reward

## How to run the Google Colab notebook:

Download and open `adaptive-probing-fl.ipynb` in Google Colab, select the A100 GPU runtime type, and run all cells.

Alternatively, use [this link](https://colab.research.google.com/github/grgarman/AdaptiveProbingFL/blob/main/adaptive-probing-fl.ipynb) to open the notebook directly in Google Colab.
