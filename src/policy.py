import os
import json
import joblib
import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, ART_MODELS, ART_REPORTS


# Costs (same as your notebook)
C_DEFAULT_APPROVED = 100
C_GOOD_REJECTED = 10
C_REVIEW = 2


def decision_from_pd(pd_value: float, t_approve: float, t_reject: float) -> str:
    if pd_value < t_approve:
        return "APPROVE"
    elif pd_value < t_reject:
        return "REVIEW"
    return "REJECT"


def total_cost(y_true, pd, t_appr, t_rej) -> float:
    y_true = np.asarray(y_true)
    pd = np.asarray(pd)

    approve = pd < t_appr
    review = (pd >= t_appr) & (pd < t_rej)
    reject = pd >= t_rej

    cost_default_approved = C_DEFAULT_APPROVED * np.sum(approve & (y_true == 1))
    cost_good_rejected = C_GOOD_REJECTED * np.sum(reject & (y_true == 0))
    cost_review = C_REVIEW * np.sum(review)

    return float(cost_default_approved + cost_good_rejected + cost_review)


def policy_outcomes(y_true, pd, t_appr, t_rej) -> dict:
    y_true = np.asarray(y_true)
    pd = np.asarray(pd)

    approve = pd < t_appr
    review = (pd >= t_appr) & (pd < t_rej)
    reject = pd >= t_rej

    return {
        "approve_rate": float(approve.mean()),
        "review_rate": float(review.mean()),
        "reject_rate": float(reject.mean()),
        "default_rate_overall": float(y_true.mean()),
        "default_rate_approved": float(y_true[approve].mean()) if approve.sum() else None,
        "default_rate_review": float(y_true[review].mean()) if review.sum() else None,
        "default_rate_reject": float(y_true[reject].mean()) if reject.sum() else None,
        "n_approved": int(approve.sum()),
        "n_review": int(review.sum()),
        "n_reject": int(reject.sum()),
        "total_cost": float(total_cost(y_true, pd, t_appr, t_rej)),
        "t_approve": float(t_appr),
        "t_reject": float(t_rej),
    }


def optimize_thresholds(y_val, pd_val, max_review_rate=None):
    t_appr_grid = np.linspace(0.05, 0.35, 31)
    t_rej_grid = np.linspace(0.20, 0.70, 51)

    best = None
    best_cost = float("inf")

    for t_appr in t_appr_grid:
        for t_rej in t_rej_grid:
            if t_rej <= t_appr:
                continue
            out = policy_outcomes(y_val, pd_val, t_appr, t_rej)
            if max_review_rate is not None and out["review_rate"] > max_review_rate:
                continue
            if out["total_cost"] < best_cost:
                best_cost = out["total_cost"]
                best = (float(t_appr), float(t_rej))

    return best, float(best_cost)


def run_policy(calibrator_file="calibrator_final.joblib", max_review_rate=None):
    X_val = pd.read_csv(DATA_PROCESSED / "X_val.csv")
    y_val = pd.read_csv(DATA_PROCESSED / "y_val.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze("columns")

    calibrator = joblib.load(ART_MODELS / calibrator_file)

    pd_val = calibrator.predict_proba(X_val)[:, 1]
    pd_test = calibrator.predict_proba(X_test)[:, 1]

    (t_appr, t_rej), _ = optimize_thresholds(y_val, pd_val, max_review_rate=max_review_rate)

    test_policy = policy_outcomes(y_test, pd_test, t_appr, t_rej)

    os.makedirs(ART_REPORTS, exist_ok=True)

    thresholds = {
        "model": calibrator_file,
        "t_approve": t_appr,
        "t_reject": t_rej,
        "costs": {
            "C_DEFAULT_APPROVED": float(C_DEFAULT_APPROVED),
            "C_GOOD_REJECTED": float(C_GOOD_REJECTED),
            "C_REVIEW": float(C_REVIEW),
        },
        "max_review_rate": float(max_review_rate) if max_review_rate is not None else None
    }

    with open(ART_REPORTS / "thresholds_final.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    with open(ART_REPORTS / "policy_summary_test.json", "w") as f:
        json.dump(test_policy, f, indent=2)

    return thresholds, test_policy


if __name__ == "__main__":
    thr, pol = run_policy()
    print("Saved thresholds_final.json and policy_summary_test.json")
    print(pol)
