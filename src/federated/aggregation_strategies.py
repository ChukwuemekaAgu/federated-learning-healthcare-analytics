"""
aggregation_strategies.py
Implements federated aggregation algorithms from scratch using NumPy.
This avoids TFF compatibility issues while demonstrating the core concepts.

Algorithms implemented:
  1. FedAvg  — McMahan et al. (2017) — weighted average of client weights
  2. FedMedian — Median aggregation (robust to Byzantine clients)
  3. FedProx  — Adds proximal term to handle non-IID data (simplified)
"""
import numpy as np
from typing import List, Tuple


def fed_avg(client_weights: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
    """
    FedAvg: weighted average of client model weights.
    Clients with more data contribute more to the global model.

    Args:
        client_weights: list of weight arrays from each client
        client_sizes:   number of training samples per client

    Returns:
        Aggregated global weights
    """
    total_samples = sum(client_sizes)
    global_weights = []

    for layer_idx in range(len(client_weights[0])):
        layer_avg = np.sum([
            (client_sizes[i] / total_samples) * client_weights[i][layer_idx]
            for i in range(len(client_weights))
        ], axis=0)
        global_weights.append(layer_avg)

    return global_weights


def fed_median(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    FedMedian: coordinate-wise median aggregation.
    More robust than FedAvg when some clients have corrupted/poisoned data.
    """
    global_weights = []

    for layer_idx in range(len(client_weights[0])):
        layer_stack = np.stack([cw[layer_idx] for cw in client_weights], axis=0)
        global_weights.append(np.median(layer_stack, axis=0))

    return global_weights


def fed_prox(
    client_weights: List[List[np.ndarray]],
    global_weights: List[np.ndarray],
    client_sizes: List[int],
    mu: float = 0.01
) -> List[np.ndarray]:
    """
    FedProx: FedAvg with proximal regularisation.
    Adds a penalty term μ/2 ||w - w_global||² to keep client updates
    close to the global model — helps with non-IID data.

    mu: proximal term coefficient (higher = more regularisation)
    """
    total_samples = sum(client_sizes)
    aggregated = []

    for layer_idx in range(len(global_weights)):
        # Weighted average of (client_update + proximal_correction)
        layer_agg = np.sum([
            (client_sizes[i] / total_samples) * (
                client_weights[i][layer_idx]
                - mu * (client_weights[i][layer_idx] - global_weights[layer_idx])
            )
            for i in range(len(client_weights))
        ], axis=0)
        aggregated.append(layer_agg)

    return aggregated


def compute_weight_divergence(
    client_weights: List[List[np.ndarray]],
    global_weights: List[np.ndarray]
) -> List[float]:
    """
    Compute L2 norm between each client's weights and the global model.
    Useful for detecting non-IID effects and Byzantine clients.
    """
    divergences = []
    for cw in client_weights:
        total_norm = sum(
            np.linalg.norm(cw[i] - global_weights[i]) ** 2
            for i in range(len(global_weights))
        ) ** 0.5
        divergences.append(total_norm)
    return divergences
