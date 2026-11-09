## About

This repository contains a working PyTorch simulator for adaptive client probing in federated learning, implemented as a Colab notebook (`adaptive-probing-fl.ipynb`), along with two baselines (Always-probe and a simplified UCB-CS) and a per-client DQN controller. The simulator supports 20 non-IID Fashion-MNIST clients, three actions per client (SKIP, HIST, PROBE), and simulated compute/communication cost accounting. Always-probe and UCB-CS run end to end and produce the expected trade-off (UCB-CS cuts total cost by ~54% for a 2.3-point test-accuracy loss), but the adaptive controller does not yet work well: it collapses to skipping all clients for roughly the first 19 of 30 rounds and reaches only ~78.5% test accuracy. Current results are means over three evaluation seeds from a single trained controller, with no per-seed curves or standard deviations saved. The immediate priorities are reward recalibration (the hand-set weights make an accuracy point worth only ~3.5 cost units, so skipping scores better than either baseline), action masking to prevent all-skip collapse, and testing shared versus difference rewards; longer training, additional baselines (Oort, random, fixed-interval probing), multi-seed reporting, and a second federation/dataset are planned next.

## How to run the Google Colab notebook:

Download and open `adaptive-probing-fl.ipynb` in Google Colab, select the A100 GPU runtime type, and run all cells.

Alternatively, use [this link](https://colab.research.google.com/github/grgarman/AdaptiveProbingFL/blob/main/adaptive-probing-fl.ipynb) to open the notebook directly in Google Colab.
