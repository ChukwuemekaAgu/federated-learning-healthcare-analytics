"""
differential_privacy.py
Implements Gaussian Mechanism Differential Privacy for federated learning.

Differential Privacy guarantees:
  - Each client's weight update is privacy-protected by adding calibrated Gaussian noise
  - The privacy budget ε (epsilon) controls privacy-accuracy trade-off:
      Low ε  → stronger privacy, more noise, lower accuracy
      High ε → weaker privacy, less noise, higher accuracy
  - δ (delta) is the probability of privacy guarantee failing (keep < 1/n)

Reference: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
"""
import numpy as np
from typing import List


def clip_gradients(weights: List[np.ndarray], clip_norm: float = 1.0) -> List[np.ndarray]:
    """
    Clip weight updates to have L2 norm ≤ clip_norm.
    This bounds the sensitivity of each client's contribution.
    """
    total_norm = sum(np.linalg.norm(w) ** 2 for w in weights) ** 0.5
    if total_norm > clip_norm:
        scale = clip_norm / (total_norm + 1e-8)
        weights = [w * scale for w in weights]
    return weights


def gaussian_mechanism(
    weights: List[np.ndarray],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    num_clients: int = 3
) -> List[np.ndarray]:
    """
    Apply Gaussian Mechanism DP noise to model weights.

    Args:
        weights:          model weight arrays after clipping
        clip_norm:        sensitivity bound (same as in clip_gradients)
        noise_multiplier: σ = noise_multiplier × clip_norm (higher → more private)
        num_clients:      used to scale noise appropriately

    Returns:
        Noised weight arrays
    """
    sigma = noise_multiplier * clip_norm
    noised = [w + np.random.normal(0, sigma, w.shape) for w in weights]
    return noised


def compute_epsilon(
    noise_multiplier: float,
    num_steps: int,
    delta: float = 1e-5,
    num_clients: int = 3
) -> float:
    """
    Approximate privacy budget ε using the moments accountant (simplified).
    For production use, use Google's dp-accounting library.

    This is an approximation for demonstration purposes.
    """
    # Simplified strong composition theorem approximation
    epsilon = (
        noise_multiplier ** -1 *
        np.sqrt(2 * num_steps * np.log(1 / delta))
    )
    return epsilon


def apply_dp_to_client_update(
    weights: List[np.ndarray],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1
) -> List[np.ndarray]:
    """
    Full DP pipeline for a single client update:
    1. Clip gradients to bound sensitivity
    2. Add calibrated Gaussian noise
    """
    clipped = clip_gradients(weights, clip_norm)
    noised = gaussian_mechanism(clipped, clip_norm, noise_multiplier)
    return noised


def dp_federated_round(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1
) -> List[np.ndarray]:
    """
    Run one DP-protected federated aggregation round.
    Each client's weights are clipped and noised before aggregation.
    """
    from federated.aggregation_strategies import fed_avg

    dp_weights = []
    for cw in client_weights:
        noised = apply_dp_to_client_update(cw, clip_norm, noise_multiplier)
        dp_weights.append(noised)

    # Aggregate with noise already applied
    global_weights = fed_avg(dp_weights, client_sizes)
    return global_weights


def privacy_report(
    noise_multiplier: float,
    num_rounds: int,
    num_samples: int,
    batch_size: int,
    delta: float = 1e-5
) -> dict:
    """Print a human-readable privacy analysis report."""
    num_steps = num_rounds * (num_samples // batch_size)
    epsilon = compute_epsilon(noise_multiplier, num_steps, delta)

    print("\n" + "="*50)
    print("  Differential Privacy Analysis")
    print("="*50)
    print(f"  Noise multiplier (σ): {noise_multiplier}")
    print(f"  Clipping norm:        1.0")
    print(f"  Communication rounds: {num_rounds}")
    print(f"  Training steps:       {num_steps}")
    print(f"  Privacy budget ε:     {epsilon:.4f}")
    print(f"  Privacy budget δ:     {delta}")
    print(f"\n  Interpretation:")
    if epsilon < 1:
        print(" Strong privacy (ε < 1)")
    elif epsilon < 10:
        print(" Moderate privacy (1 ≤ ε < 10)")
    else:
        print(" Weak privacy (ε ≥ 10) — increase noise_multiplier")
    print("="*50)

    return {"epsilon": epsilon, "delta": delta, "noise_multiplier": noise_multiplier}


if __name__ == "__main__":
    privacy_report(
        noise_multiplier=1.1,
        num_rounds=10,
        num_samples=1500,
        batch_size=32
    )
