import os
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .config import DATA_PROCESSED, ART_MODELS, ART_REPORTS, ART_METRICS, RANDOM_STATE
from .metrics import eval_proba


def load_processed():
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_val = pd.read_csv(DATA_PROCESSED / "X_val.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(DATA_PROCESSED / "y_val.csv").squeeze("columns")
    return X_train, X_val, y_train, y_val


def train_baselines():
    X_train, X_val, y_train, y_val = load_processed()

    models = [
        ("LogReg", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        ("HistGB", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.07,
            max_iter=300,
            random_state=RANDOM_STATE
        )),
    ]

    results = []
    fitted = {}

    for name, model in models:
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        metrics = {"model": name, **eval_proba(y_val, prob_val)}
        results.append(metrics)
        fitted[name] = model

    leaderboard = pd.DataFrame(results).sort_values(
        by=["roc_auc", "log_loss", "ece_10bin"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    winner_name = leaderboard.loc[0, "model"]
    winner_model = fitted[winner_name]

    os.makedirs(ART_MODELS, exist_ok=True)
    os.makedirs(ART_REPORTS, exist_ok=True)
    os.makedirs(ART_METRICS, exist_ok=True)

    leaderboard.to_csv(ART_REPORTS / "leaderboard.csv", index=False)
    joblib.dump(winner_model, ART_MODELS / f"best_model_{winner_name}.joblib")

    run_summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "winner": winner_name,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    with open(ART_METRICS / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    return winner_name, winner_model


if __name__ == "__main__":
    wname, _ = train_baselines()
    print("Winner saved:", wname)
