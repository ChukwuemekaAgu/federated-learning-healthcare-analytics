"""
utils/data_utils.py
Shared data utilities for federated learning experiments.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


FEATURE_COLS = [
    "patient_age", "gender", "diagnosis_code", "procedure_code",
    "hospital_days", "claim_amount", "num_previous_claims",
    "provider_type", "region", "insurance_plan"
]
TARGET_COL = "unnecessary_claim"


def load_and_normalize(path: str) -> tuple:
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    X_min, X_max = X.min(0, keepdims=True), X.max(0, keepdims=True) + 1e-8
    X = (X - X_min) / (X_max - X_min)
    return X, y


def plot_training_history(history_csv: str, save_path: str = None) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["round"], df["global_loss"], marker="o", label="Global Loss")
    for col in [c for c in df.columns if "client" in c and "loss" in c]:
        axes[0].plot(df["round"], df[col], linestyle="--", alpha=0.6, label=col)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per Round")
    axes[0].legend()

    axes[1].plot(df["round"], df["global_accuracy"], marker="o", color="green", label="Accuracy")
    if "global_auc" in df.columns:
        axes[1].plot(df["round"], df["global_auc"], marker="s", color="orange", label="AUC")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Global Model Performance")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Plot saved → {save_path}")
    else:
        plt.savefig("results/fl_training_history.png", dpi=150)
    plt.close()


def compare_fl_vs_centralized(fl_csv: str, central_csv: str, save_path: str = None) -> None:
    fl = pd.read_csv(fl_csv)
    central = pd.read_csv(central_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fl["round"], fl["global_accuracy"], marker="o", label="Federated (FedAvg)")
    ax.axhline(
        central["val_accuracy"].max(),
        color="red", linestyle="--",
        label=f"Centralised Best ({central['val_accuracy'].max():.4f})"
    )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Federated vs Centralised Accuracy")
    ax.legend()
    plt.tight_layout()

    out = save_path or "results/fl_vs_centralized.png"
    plt.savefig(out, dpi=150)
    print(f"[INFO] Comparison plot saved → {out}")
    plt.close()
