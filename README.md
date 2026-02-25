# Federated Learning — Healthcare Analytics

> **Project:** Privacy-Preserving Multi-Site Healthcare Claims Analysis
> **Key technique:** Federated Averaging (FedAvg) + Differential Privacy
> **No TensorFlow Federated required** — implemented from scratch with TensorFlow/Keras + NumPy

---

## Overview

This project implements federated learning for healthcare claims analysis across **3 simulated hospital clients**. Instead of pooling sensitive patient data into a central server, each hospital trains locally and only shares model weight updates — preserving patient privacy while enabling collaborative learning.

This is the privacy-first companion to the [centralised ML repo](https://github.com/ChukwuemekaAgu/privacy-preserving-healthcare-claims-ml).

---

## Repository Structure

```
federated-learning-healthcare-analytics/
│
├── data/
│   ├── client_1/claims.csv       ← Hospital A (generated)
│   ├── client_2/claims.csv       ← Hospital B (generated)
│   ├── client_3/claims.csv       ← Hospital C (generated)
│   ├── generate_clients.py       ← Script to generate client data
│   └── README.md
│
├── src/
│   ├── federated/
│   │   ├── fl_model.py            ← Keras model definition
│   │   ├── federated_training.py  ← Main FL training loop (FedAvg)
│   │   └── aggregation_strategies.py ← FedAvg, FedMedian, FedProx
│   │
│   ├── centralized/
│   │   └── centralized_baseline.py ← Same model on pooled data
│   │
│   ├── privacy/
│   │   ├── differential_privacy.py ← Gaussian mechanism DP
│   │   └── threat_model.md         ← Privacy threat analysis
│   │
│   └── utils/
│       └── data_utils.py           ← Plotting and data helpers
│
├── experiments/
│   ├── convergence_analysis.ipynb
│   ├── non_iid_data.ipynb
│   └── privacy_vs_accuracy.ipynb
│
├── results/                        ← Auto-generated
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ChukwuemekaAgu/federated-learning-healthcare-analytics.git
cd federated-learning-healthcare-analytics

# 2. Create & activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate client datasets
python data/generate_clients.py

# 5. Run federated training
python src/federated/federated_training.py

# 6. Run centralised baseline (for comparison)
python src/centralized/centralized_baseline.py
```

---

## Key Concepts

### Federated Averaging (FedAvg)
Each communication round:
1. Server sends global weights to all clients
2. Each client trains locally for 5 epochs
3. Clients send updated weights back to server
4. Server computes weighted average → new global model

### Aggregation Strategies
| Strategy | When to use |
|----------|------------|
| FedAvg | Default — weighted by dataset size |
| FedMedian | When some clients may be compromised |
| FedProx | When client data is highly non-IID |

### Differential Privacy
- Gaussian noise added to weight updates before transmission
- Gradient clipping bounds each client's sensitivity
- Privacy budget ε tracks total privacy expenditure

---

## Are the Two Repos Connected?

These two repositories are **independent but complementary**:

| Aspect | This Repo | Centralized Repo |
|--------|-----------|-----------------|
| Data location | Stays on each client | Pooled centrally |
| Privacy | Formal DP guarantees | Anonymisation only |
| Model training | Distributed | Centralized |
| Accuracy | Slightly lower | Slightly higher |
| Real-world fit | Multi-hospital deployment | Single-institution |

**You can run either repo without the other.** The centralized repo demonstrates the accuracy ceiling; this repo shows what's achievable with privacy guarantees.

---

## References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Li et al. (2020) — [FedProx: Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- Dwork & Roth (2014) — The Algorithmic Foundations of Differential Privacy
