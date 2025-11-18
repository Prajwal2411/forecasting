"""Training script for the internal-fill classifier and time-to-fill regressor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

FEATURE_COLUMNS = [
    "team_size_required",
    "budget_amount",
    "urgency_score",
    "seniority_encoded",
    "avg_skill_match",
    "historical_internal_fill_rate",
]


def build_training_frame(data_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    employees = pd.read_csv(data_dir / "employees.csv")
    projects = pd.read_csv(data_dir / "projects.csv")
    assignments = pd.read_csv(data_dir / "assignments.csv")

    role_fill = (
        assignments.groupby("role_id")["billable_flag"]
        .apply(lambda x: (x == "Y").mean())
        .to_dict()
    )

    rows = []
    for idx in range(400):
        emp = employees.sample(1, random_state=seed + idx).iloc[0]
        proj = projects.sample(1, random_state=seed + idx + 1).iloc[0]
        urgency_map = {"low": 1, "medium": 2, "high": 3}
        urgency_score = urgency_map.get(str(proj.get("urgency", "medium")).lower(), 2)
        team_size = int(rng.integers(3, 11))
        match = float(
            rng.uniform(0.4, 0.95)
            if emp["primary_role_id"] in proj.get("project_type", "")
            else rng.uniform(0.2, 0.7)
        )
        historical_rate = role_fill.get(emp["primary_role_id"], rng.uniform(0.3, 0.7))
        seniority = {"Junior": 1, "Mid": 2, "Senior": 3}.get(emp["seniority_level"], 2)
        base_prob = 0.3 + 0.3 * match + 0.1 * (historical_rate - 0.5)
        prob_internal = min(0.95, max(0.05, base_prob - 0.05 * urgency_score))
        internal_fill = rng.random() < prob_internal
        time_to_fill = (
            25 + 10 * urgency_score + 15 * (3 - seniority) - 10 * match + rng.normal(0, 3)
        )
        rows.append(
            {
                "team_size_required": team_size,
                "budget_amount": float(proj["budget_amount"]),
                "urgency_score": urgency_score,
                "seniority_encoded": seniority,
                "avg_skill_match": match,
                "historical_internal_fill_rate": historical_rate,
                "internal_fill": int(internal_fill),
                "time_to_fill_days": max(10, float(time_to_fill)),
            }
        )
    return pd.DataFrame(rows)


def build_model(input_dim: int, task: str) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid" if task == "classification" else "linear"),
        ]
    )
    loss = "binary_crossentropy" if task == "classification" else "mse"
    metric = ["accuracy"] if task == "classification" else ["mae"]
    model.compile(optimizer="adam", loss=loss, metrics=metric)
    return model


def train_models(data_dir: Path, model_dir: Path, seed: int) -> None:
    df = build_training_frame(data_dir, seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y_class = df["internal_fill"].values
    y_reg = df["time_to_fill_days"].values

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"columns": FEATURE_COLUMNS, "scaler": scaler}, model_dir / "preprocessing.joblib")

    classifier = build_model(X.shape[1], task="classification")
    classifier.fit(X, y_class, epochs=10, batch_size=32, verbose=0)
    classifier.save(model_dir / "internal_fill.keras")

    regressor = build_model(X.shape[1], task="regression")
    regressor.fit(X, y_reg, epochs=12, batch_size=32, verbose=0)
    regressor.save(model_dir / "time_to_fill.keras")

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "seed": seed,
        "num_rows": len(df),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HR forecasting MLP models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/registry/v1"))
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(args.data_dir, args.model_dir, args.seed)


if __name__ == "__main__":
    main()
