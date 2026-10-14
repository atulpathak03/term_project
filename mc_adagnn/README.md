# Fed-ADAGNN: Federated Regional Short-Term Load Forecasting

This repository contains the PyTorch implementation of the **MC-ADAGNN** (Multi-Channel Fusion and Adaptive Graph Neural Network) architecture for regional short-term load forecasting. 

As a novel extension for Phase 2 research, this codebase introduces **Fed-ADAGNN**, a Privacy-Preserving Federated Learning framework that allows multiple independent grids (clients) to collaboratively train the forecasting model without exposing their highly sensitive, high-resolution raw power consumption data.

## Features
- **Adaptive Graph Construction**: Dynamically learns inter-industry dependencies without predefined adjacency matrices using Skeleton-Foundation, Core-Stability, and Dynamic-Interaction channels.
- **Mix-hop Graph Convolution**: Captures long-range spatiotemporal dependencies.
- **Inception-Dilated Temporal Convolution**: Captures multi-scale periodic variation patterns (e.g., daily, weekly).
- **Federated Learning Support**: Decentralized training across multiple clients using the Federated Averaging (FedAvg) algorithm.
- **IEEE 14-bus System Data Generation**: Built-in scripts to simulate complex regional load patterns based on the IEEE 14-bus standard system.
- **False Data Injection Attack (FDIA) Simulation**: Evaluates model robustness against malicious scaling, random, or targeted attacks.

## File Structure

```
mc_adagnn/
├── model.py                # Core PyTorch implementation of MC-ADAGNN architecture and Loss Function
├── dataset.py              # Dataset generator and DataLoader for centralized training (Phase 1)
├── train.py                # Training script for the baseline centralized MC-ADAGNN model
├── fed_dataset.py          # Dataset generator for multi-client federated environments
├── fed_train.py            # Federated Learning loop (FedAvg) training script (Phase 2)
├── generate_ieee14.py      # Script to generate IEEE 14-bus standard system load data
├── fdia.py                 # Core logic for generating False Data Injection Attacks
├── train_fdia.py           # Script for centralized FDIA robustness experiments
├── fed_fdia_train.py       # Script for federated FDIA robustness experiments
├── utils_eval.py           # Evaluation metrics utilities
├── utils_vis.py            # Visualization utilities for plotting metrics and results
└── data/                   # Auto-generated directory containing the synthetic/IEEE14 CSV data
```

## Prerequisites

- Python 3.8+
- PyTorch (2.0+)
- Pandas
- Numpy
- pandapower (for IEEE 14-bus generation)

## Usage

### 1. Centralized Baseline (Phase 1)
To generate a centralized dataset and train the baseline MC-ADAGNN model:
```bash
python train.py --epochs 10 --batch_size 16 --lr 0.001
```
*Note: The dataset (`data/synthetic_load_data.csv`) is automatically generated if it does not exist.*

### 2. Federated Learning (Phase 2 - Novel Extension)
To simulate a federated environment (e.g., 3 independent cities/clients) and collaboratively train the global model:
```bash
python fed_train.py --num_clients 3 --rounds 10 --local_epochs 2
```
*Note: Client-specific datasets will be generated under `data/federated/` automatically.*

## Hyperparameters
You can customize the architecture and training process via command-line arguments:
- `--in_seq_len`: Input sequence length (default: 168 hours / 7 days).
- `--out_seq_len`: Output forecasting sequence length (default: 24 hours).
- `--embed_dim`: Dimension for node embeddings in the Skeleton-Foundation channel.
- `--hidden_channels`: Number of hidden channels in SpatioTemporal layers.
- `--num_layers`: Number of SpatioTemporal layers.
- `--gamma`: Weight for the stability loss factor.

## Research & Conference Paper Application
This repository serves as the empirical foundation for researching the transition of spatiotemporal GNN forecasters from centralized data lakes to decentralized, privacy-preserving grid edge devices. The architecture proves that multi-channel spatiotemporal correlations can be learned collaboratively through gradient sharing, maintaining baseline accuracy without compromising industrial data privacy.

## False Data Injection Attack (FDIA) Experiments

This module adds a simulation environment to evaluate the robustness of both MC-ADAGNN and Fed-ADAGNN under False Data Injection Attacks. FDIA represents malicious modifications to regional load measurements (e.g. smart meter tampering) which can corrupt forecasting results or federated global models.

### Centralized FDIA Experiments
To test how a clean-trained MC-ADAGNN model behaves under varying attack strengths, you can run the automated matrix:
```bash
python train_fdia.py --run_matrix --epochs 5
```
This trains the baseline model and tests it against clean inputs, and then against inputs corrupted by a simple scaling attack (`x_attacked = x * (1 + alpha)`) with strengths like 5%, 10%, and 20%.

To run a specific attack manually:
```bash
python train_fdia.py --attack scaling --attack_strength 0.10
```

### Federated FDIA Experiments
In a federated setting, you can simulate specific clients being compromised (e.g., Client 2 out of 3):
```bash
python fed_fdia_train.py --num_clients 3 --rounds 10 --local_epochs 2 --attacked_clients 2 --attack scaling --attack_strength 0.10
```
This simulates a data-poisoning attack where the compromised client applies FDIA to its local measurements before training and computing its local update.

### Evaluation and Metrics
Experiments do NOT attack the ground truth labels. This ensures we are measuring the true degradation of forecasting capabilities caused by compromised input measurements.
Results and visualizations are stored in the `results/` directory:
- `centralized_fdia.csv`, `federated_fdia.csv`, `round_metrics.csv`
- Plot visualizations (Actual vs Predicted, RMSE over rounds, Degradation) are saved in `results/plots/`.

