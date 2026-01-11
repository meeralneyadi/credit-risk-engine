import os
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .config import DATA_PROCESSED, ART_MODELS, ART_METRICS
from .metrics import eval_proba


def load_splits():
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_val = pd.read_csv(DATA_PROCESSED / "X_val.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")

    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(DATA_PROCESSED / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze("columns")
    return X_train, X_val, X_test, y_train, y_val, y_test


def calibrate_from_model(model_path: str, out_name: str = "calibrator_final.joblib"):
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    base_model = joblib.load(model_path)

    # Fit on TRAIN
    base_model.fit(X_train, y_train)

    # Calibrate on VAL (prefit)
    calibrator = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)

    # Evaluate on TEST
    pd_test = calibrator.predict_proba(X_test)[:, 1]
    test_metrics = eval_proba(y_test, pd_test)

    os.makedirs(ART_MODELS, exist_ok=True)
    os.makedirs(ART_METRICS, exist_ok=True)

    joblib.dump(calibrator, ART_MODELS / out_name)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "calibrator_file": out_name,
        "base_model_file": str(model_path),
        "test_metrics": test_metrics
    }
    with open(ART_METRICS / "calibration_test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    # default: calibrate the HistGB tuned model if it exists, else best winner
    tuned = ART_MODELS / "best_tuned_model_HistGB_Tuned.joblib"
    if tuned.exists():
        mpath = str(tuned)
    else:
        # fallback example (change to your actual winner file if needed)
        # e.g. artifacts/models/best_model_HistGB.joblib
        candidates = list(ART_MODELS.glob("best_model_*.joblib"))
        if not candidates:
            raise FileNotFoundError("No best_model_*.joblib found. Run src/train.py first.")
        mpath = str(candidates[0])

    metrics = calibrate_from_model(mpath)
    print("Calibration TEST metrics:", metrics)
