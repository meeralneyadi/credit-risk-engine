from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Credit Risk Engine", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
ART_MODELS = ROOT / "artifacts" / "models"
ART_REPORTS = ROOT / "artifacts" / "reports"
DATA_PROCESSED = ROOT / "data" / "processed"

CALIBRATOR_PATH = ART_MODELS / "calibrator_final.joblib"
THRESHOLDS_PATH = ART_REPORTS / "thresholds_final.json"
FEATURES_PATH = DATA_PROCESSED / "X_train.csv"


def decision_from_pd(pd_value: float, t_approve: float, t_reject: float) -> str:
    if pd_value < t_approve:
        return "APPROVE"
    elif pd_value < t_reject:
        return "REVIEW"
    return "REJECT"


@app.on_event("startup")
def load_artifacts():
    global calibrator, thresholds, feature_columns

    if not CALIBRATOR_PATH.exists():
        raise RuntimeError(f"Missing calibrator: {CALIBRATOR_PATH}")

    if not THRESHOLDS_PATH.exists():
        raise RuntimeError(f"Missing thresholds: {THRESHOLDS_PATH}")

    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Missing features reference file: {FEATURES_PATH}")

    calibrator = joblib.load(CALIBRATOR_PATH)

    with open(THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)

    feature_columns = list(pd.read_csv(FEATURES_PATH, nrows=1).columns)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    feats = req.features
    missing = [c for c in feature_columns if c not in feats]
    extra = [k for k in feats.keys() if k not in feature_columns]

    # Fill missing with 0.0 (safe default). You can change to mean later.
    row = {c: float(feats.get(c, 0.0)) for c in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    try:
        pd_value = float(calibrator.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {type(e).__name__}: {e}")

    t_approve = float(thresholds["t_approve"])
    t_reject = float(thresholds["t_reject"])

    decision = decision_from_pd(pd_value, t_approve, t_reject)

    return PredictResponse(
        pd=pd_value,
        decision=decision,
        thresholds={"t_approve": t_approve, "t_reject": t_reject},
        missing_features=missing,
        extra_features=extra,
    )
