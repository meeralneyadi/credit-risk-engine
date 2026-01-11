# Credit Risk Engine
End-to-End Machine Learning System for Credit Decisioning

---

## Overview
This project implements a production-style credit risk engine that predicts the Probability of Default (PD) for credit applicants and converts it into business decisions:

- APPROVE  
- REVIEW  
- REJECT  

Unlike basic classification projects, this system focuses on real-world credit risk practices including probability calibration, cost-sensitive decision making, and deployable APIs.

---

## Key Features
- Multiple machine learning models benchmarked (Logistic Regression, Random Forest, Gradient Boosting)
- Automatic model selection based on discrimination and calibration metrics
- Isotonic calibration for reliable probability of default estimates
- Cost-based decision threshold optimization
- Strict train / validation / test separation (no data leakage)
- FastAPI service with interactive Swagger documentation
- Reusable core engine code in `src/` (not notebook-only)
- Policy analytics including approval and default rates per bucket

---

## Project Structure

    credit-risk-engine/
    ├── api/                    # FastAPI application
    │   ├── app.py
    │   └── schemas.py
    │
    ├── src/                    # Core ML and business logic
    │   ├── data_prep.py
    │   ├── train.py
    │   ├── calibrate.py
    │   ├── policy.py
    │   ├── metrics.py
    │   └── config.py
    │
    ├── notebooks/              # EDA and experimentation
    │   ├── 01_eda_application.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_training_leaderboard.ipynb
    │   └── 04_decision_engine_demo.ipynb
    │
    ├── data/                   # Local only (ignored by Git)
    ├── artifacts/              # Models and reports (ignored by Git)
    │
    ├── README.md
    ├── requirements.txt
    └── .gitignore

---

## Dataset
- Source: UCI – Default of Credit Card Clients
- Size: approximately 30,000 applicants
- Target: binary default indicator
- Default rate: approximately 22%

The dataset contains demographic, financial, and repayment history features suitable for classical credit risk modeling.

Raw and processed data are intentionally excluded from version control.

---

## Modeling and Evaluation

### Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- Histogram Gradient Boosting (selected model)

### Metrics Used
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score
- KS Statistic
- Expected Calibration Error (ECE)

### Calibration
The final model is calibrated using Isotonic Regression:
- Base model trained on the training set
- Calibration learned on the validation set
- Final performance evaluated on the test set

This ensures probability outputs are suitable for real-world decision making.

---

## Decision Engine
Predicted probabilities of default are converted into actions using cost-sensitive optimization.

| Decision | Description |
|--------|-------------|
| APPROVE | Low predicted risk |
| REVIEW  | Medium risk requiring manual underwriting |
| REJECT  | High predicted risk |

Thresholds are selected by minimizing total expected cost on the validation set and evaluated on the test set.

---

## API (FastAPI)

### Run the API
```bash
python -m uvicorn api.app:app --reload
Endpoints
GET /health – service health check
POST /predict – predict probability of default and decision
Example Request
{
  "features": {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0
  }
}
Example Response
{
  "pd": 0.18,
  "decision": "REVIEW",
  "thresholds": {
    "t_approve": 0.05,
    "t_reject": 0.63
  }
}
Reproduce Results
From the project root:
python -m src.data_prep
python -m src.train
python -m src.calibrate
python -m src.policy
Design Principles
No data leakage
Separation of modeling and business logic
Reproducible artifacts
API-first design
Realistic credit risk assumptions
Future Improvements
Review capacity constraints
Batch prediction endpoint
SHAP-based explanations
Model monitoring and drift detection
CI/CD for retraining pipelines
Author
Meera AlNeyadi
Computer Science and Software Engineering
Machine Learning, Data, Systems

---

## FINAL STEPS (LAST TIME I PROMISE)
```bash
git add README.md
git commit -m "Fix README formatting and structure"
git push
