"""
federated_training.py
Federated Learning training loop for healthcare claims analysis.

Implements FedAvg from scratch using TensorFlow/Keras + NumPy.
No TensorFlow Federated (TFF) required — avoids all TFF version issues.

Architecture:
  - Global server holds the global model
  - Each round: clients train locally → send weights → server aggregates
  - Repeat for N communication rounds
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.fl_model import create_keras_model, get_model_weights, set_model_weights
from federated.aggregation_strategies import fed_avg, fed_median, fed_prox, compute_weight_divergence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# ── Configuration ──────────────────────────────────────────────────────────────
NUM_ROUNDS = 10          # communication rounds
LOCAL_EPOCHS = 5         # local epochs per client per round
BATCH_SIZE = 32
LEARNING_RATE = 0.01
AGGREGATION = "fedavg"   # "fedavg" | "fedmedian" | "fedprox"

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
INPUT_DIM = 10

FEATURE_COLS = [
    "patient_age", "gender", "diagnosis_code", "procedure_code",
    "hospital_days", "claim_amount", "num_previous_claims",
    "provider_type", "region", "insurance_plan"
]
TARGET_COL = "unnecessary_claim"


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_client_data(client_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalise a single client's claims data."""
    path = DATA_DIR / f"client_{client_id}" / "claims.csv"
    df = pd.read_csv(path)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Min-max normalise per client (in practice, use global stats)
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True) + 1e-8
    X = (X - X_min) / (X_max - X_min)

    return X, y


def load_all_clients(num_clients: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    clients = []
    for i in range(1, num_clients + 1):
        X, y = load_client_data(i)
        clients.append((X, y))
        print(f"[INFO] Client {i}: {len(X)} samples | "
              f"Unnecessary: {int(y.sum())} ({y.mean()*100:.1f}%)")
    return clients


# ── Client Training ────────────────────────────────────────────────────────────
def train_client(
    global_weights: list,
    X: np.ndarray,
    y: np.ndarray,
    local_epochs: int = LOCAL_EPOCHS
) -> Tuple[list, dict]:
    """
    Train a local model starting from global weights.
    Returns updated weights and training metrics.
    """
    model = create_keras_model(INPUT_DIM, LEARNING_RATE)
    set_model_weights(model, global_weights)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(BATCH_SIZE)

    history = model.fit(dataset, epochs=local_epochs, verbose=0)

    metrics = {
        "loss": history.history["loss"][-1],
        "accuracy": history.history["accuracy"][-1],
    }
    return get_model_weights(model), metrics


# ── Server Aggregation ────────────────────────────────────────────────────────
def aggregate(
    client_weights: List[list],
    global_weights: list,
    client_sizes: List[int],
    strategy: str = AGGREGATION
) -> list:
    if strategy == "fedavg":
        return fed_avg(client_weights, client_sizes)
    elif strategy == "fedmedian":
        return fed_median(client_weights)
    elif strategy == "fedprox":
        return fed_prox(client_weights, global_weights, client_sizes)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


# ── Global Evaluation ─────────────────────────────────────────────────────────
def evaluate_global(global_weights: list, clients: List[Tuple]) -> dict:
    """Evaluate global model on all clients' data combined."""
    all_X = np.concatenate([c[0] for c in clients], axis=0)
    all_y = np.concatenate([c[1] for c in clients], axis=0)

    model = create_keras_model(INPUT_DIM)
    set_model_weights(model, global_weights)

    results = model.evaluate(all_X, all_y, verbose=0)
    metric_names = model.metrics_names
    return {name: val for name, val in zip(metric_names, results)}


# ── Main Federated Training Loop ──────────────────────────────────────────────
def run_federated_training():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Federated Learning — Healthcare Claims Analytics")
    print(f"  Strategy: {AGGREGATION.upper()} | Rounds: {NUM_ROUNDS}")
    print("="*60)

    # Load all client data
    print("\n[STEP 1] Loading client datasets...")
    clients = load_all_clients(num_clients=3)
    client_sizes = [len(c[0]) for c in clients]

    # Initialise global model
    global_model = create_keras_model(INPUT_DIM, LEARNING_RATE)
    global_weights = get_model_weights(global_model)

    # Training history for logging
    history = []

    print(f"\n[STEP 2] Starting federated training ({NUM_ROUNDS} rounds)...")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")

        # Each client trains locally
        client_updated_weights = []
        client_metrics = []

        for i, (X, y) in enumerate(clients):
            updated_weights, metrics = train_client(global_weights, X, y)
            client_updated_weights.append(updated_weights)
            client_metrics.append(metrics)
            print(f"  Client {i+1}: loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f}")

        # Server aggregates
        global_weights = aggregate(
            client_updated_weights, global_weights, client_sizes, AGGREGATION
        )

        # Evaluate global model
        global_metrics = evaluate_global(global_weights, clients)
        print(f"  Global → loss={global_metrics['loss']:.4f} | acc={global_metrics['accuracy']:.4f} | auc={global_metrics['auc']:.4f}")

        # Track divergence between clients (non-IID analysis)
        divergences = compute_weight_divergence(client_updated_weights, global_weights)
        print(f"  Weight divergences: {[f'{d:.4f}' for d in divergences]}")

        history.append({
            "round": round_num,
            **{f"global_{k}": v for k, v in global_metrics.items()},
            **{f"client_{i+1}_loss": client_metrics[i]["loss"] for i in range(3)}
        })

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_path = RESULTS_DIR / "federated_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"\n[INFO] Training history saved → {hist_path}")

    # Save final global model weights
    np.save(str(RESULTS_DIR / "global_model_weights.npy"), np.array(global_weights, dtype=object))
    print(f"[INFO] Global model weights saved → {RESULTS_DIR}/global_model_weights.npy")

    print("\n✅ Federated training complete!")
    print(f"   Final global accuracy: {history[-1]['global_accuracy']:.4f}")
    print(f"   Final global AUC:      {history[-1]['global_auc']:.4f}")

    return history, global_weights


if __name__ == "__main__":
    run_federated_training()
