"""
generate_clients.py
Generates synthetic healthcare claims data for 3 simulated hospital clients.
Each client has slightly different data distributions (non-IID) to simulate
real-world federated learning conditions.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_client_data(client_id: int, n_samples: int = 500) -> pd.DataFrame:
    """
    Each client (hospital) has slightly different patient demographics
    and claim patterns — simulating non-IID federated data.
    """
    # Vary distributions per client to simulate different hospitals
    age_mean = {1: 45, 2: 55, 3: 38}[client_id]
    cost_scale = {1: 120_000, 2: 180_000, 3: 90_000}[client_id]
    fraud_rate = {1: 0.08, 2: 0.12, 3: 0.06}[client_id]

    df = pd.DataFrame({
        "patient_age": np.clip(np.random.normal(age_mean, 12, n_samples), 18, 85).astype(int),
        "gender": np.random.choice([0, 1], n_samples),  # 0=M, 1=F
        "diagnosis_code": np.random.randint(0, 5, n_samples),
        "procedure_code": np.random.randint(0, 5, n_samples),
        "hospital_days": np.random.randint(0, 15, n_samples),
        "claim_amount": np.clip(
            np.random.exponential(scale=cost_scale, size=n_samples), 5_000, 1_000_000
        ).round(2),
        "num_previous_claims": np.random.randint(0, 20, n_samples),
        "provider_type": np.random.randint(0, 3, n_samples),
        "region": np.random.randint(0, 4, n_samples),
        "insurance_plan": np.random.randint(0, 3, n_samples),
    })

    # Cost features
    df["cost_per_day"] = df["claim_amount"] / (df["hospital_days"] + 1)

    # Target: unnecessary claim
    df["unnecessary_claim"] = (
        (df["claim_amount"] > df["claim_amount"].quantile(1 - fraud_rate)) &
        (df["hospital_days"] < 3)
    ).astype(int)

    return df


def main():
    base_dir = Path(__file__).parent

    for client_id in [1, 2, 3]:
        client_dir = base_dir / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)

        df = generate_client_data(client_id, n_samples=500)
        output_path = client_dir / "claims.csv"
        df.to_csv(output_path, index=False)

        print(f"[Client {client_id}] {len(df)} rows → {output_path}")
        print(f"           Unnecessary claims: {df['unnecessary_claim'].sum()} ({df['unnecessary_claim'].mean()*100:.1f}%)")

    print("\n✅ All client datasets generated.")


if __name__ == "__main__":
    main()
