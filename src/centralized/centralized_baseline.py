"""
centralized_baseline.py
Trains the same neural network architecture on pooled data from all clients.
Used to compare federated vs centralised performance.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.fl_model import create_keras_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
INPUT_DIM = 10

FEATURE_COLS = [
    "patient_age", "gender", "diagnosis_code", "procedure_code",
    "hospital_days", "claim_amount", "num_previous_claims",
    "provider_type", "region", "insurance_plan"
]
TARGET_COL = "unnecessary_claim"


def load_pooled_data():
    """Pool all client data — simulates a centralised scenario."""
    dfs = []
    for i in [1, 2, 3]:
        path = DATA_DIR / f"client_{i}" / "claims.csv"
        df = pd.read_csv(path)
        df["source_client"] = i
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Pooled data: {len(df)} rows from {df['source_client'].nunique()} clients")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    X_min, X_max = X.min(0), X.max(0) + 1e-8
    X = (X - X_min) / (X_max - X_min)

    return X, y


def train_centralized(epochs: int = 30, batch_size: int = 32):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Centralized Baseline — Healthcare Claims")
    print("="*60)

    X, y = load_pooled_data()

    # 80/20 split
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    model = create_keras_model(INPUT_DIM)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    final = {name: val for name, val in zip(metric_names, results)}

    print(f"\n[RESULT] Centralized model:")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_path = RESULTS_DIR / "centralized_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"[INFO] History saved → {hist_path}")

    model.save(str(RESULTS_DIR / "centralized_model.keras"))
    print(f"[INFO] Model saved → {RESULTS_DIR}/centralized_model.keras")

    return final


if __name__ == "__main__":
    train_centralized()
