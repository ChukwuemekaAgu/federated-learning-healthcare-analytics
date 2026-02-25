# Threat Model — Federated Learning for Healthcare Claims

## Overview

This document outlines the privacy threats considered in this federated learning system and the mitigations applied.

---

## System Actors

| Actor | Role | Trust Level |
|-------|------|-------------|
| Hospital Clients (1–3) | Train local models on patient claims | Semi-trusted |
| Central Server (Aggregator) | Aggregates model weights | Honest-but-curious |
| External Adversary | No direct access | Untrusted |

---

## Threat Categories

### 1. Honest-but-Curious Server
**Threat:** The aggregating server follows the protocol correctly but attempts to infer private patient information from the model weight updates received from clients.

**Mitigation:** Differential Privacy (Gaussian Mechanism) is applied to all client weight updates before transmission. The server only ever sees noised weights.

### 2. Model Inversion Attacks
**Threat:** An adversary reconstructs training data from model gradients or weights — known as gradient inversion.

**Mitigation:**
- Gradient clipping bounds the information any single update can reveal
- Gaussian noise addition (DP) degrades reconstructed data quality
- Local batch training (multiple local epochs) reduces gradient leakage vs. single-step SGD

### 3. Membership Inference Attacks
**Threat:** An adversary determines whether a specific patient's record was in a client's training set by querying the model.

**Mitigation:**
- DP-SGD with (ε, δ)-DP provides formal membership inference resistance
- Regularisation (Dropout, L2) reduces overfitting that enables membership inference

### 4. Byzantine / Poisoning Attacks
**Threat:** A malicious client sends corrupted model updates to degrade the global model or insert backdoors.

**Mitigation:**
- FedMedian aggregation is robust to up to 50% malicious clients
- Weight divergence monitoring (`compute_weight_divergence`) flags anomalous updates

### 5. Data Reconstruction from Aggregated Model
**Threat:** Reverse-engineering the final global model to extract training data.

**Mitigation:**
- The global model is only released after post-processing
- Access to the global model is controlled at the application level

---

## Privacy Parameters Used

| Parameter | Value | Justification |
|-----------|-------|---------------|
| ε (epsilon) | ~2–8 | Balances accuracy with strong privacy |
| δ (delta) | 1e-5 | Below 1/N (N = total samples ≈ 1500) |
| Clipping norm | 1.0 | Standard value from DP-SGD literature |
| Noise multiplier | 1.1 | Provides ε < 5 over 10 rounds |

---

## Assumptions and Limitations

- This implementation uses a **simplified moments accountant**. For production, use Google's `dp-accounting` library for accurate ε computation.
- The threat model assumes clients do not collude with each other.
- Physical security of client hospital servers is outside the scope of this model.

---

## References

- McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data
- Dwork & Roth (2014) — The Algorithmic Foundations of Differential Privacy
- Bonawitz et al. (2017) — Practical Secure Aggregation for Privacy-Preserving Machine Learning
- Blanchard et al. (2017) — Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (FedMedian)
