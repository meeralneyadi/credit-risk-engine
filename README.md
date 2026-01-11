💰 Credit Risk Engine
End-to-End Machine Learning System for Credit Decisioning
Overview
This project implements a production-style credit risk engine that predicts the probability of default (PD) for credit card applicants and converts it into business decisions:
✅ APPROVE
🟡 REVIEW
❌ REJECT
Unlike simple classification projects, this system focuses on:
Probability calibration
Cost-sensitive decision making
Train / Validation / Test discipline
Deployable API design
The project follows real-world credit risk practices used in banks and fintech companies.
Key Features
Multiple ML models benchmarked (Logistic Regression, Random Forest, Gradient Boosting)
Model selection based on discrimination + calibration
Isotonic calibration for reliable PD estimates
Cost-based threshold optimization
Strict train / validation / test separation (no leakage)
FastAPI service with Swagger UI
Reusable src/ code (not notebook-only)
Policy analytics (approval rates, default rates per bucket)
Project Structure
credit-risk-engine/
│
├── api/                    # FastAPI service
│   ├── app.py
│   └── schemas.py
│
├── src/                    # Core ML & business logic
│   ├── data_prep.py
│   ├── train.py
│   ├── calibrate.py
│   ├── policy.py
│   ├── metrics.py
│   └── config.py
│
├── notebooks/              # EDA & experimentation
│   ├── 01_eda_application.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_leaderboard.ipynb
│   └── 04_decision_engine_demo.ipynb
│
├── artifacts/              # Models, thresholds, reports
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
Dataset
Source: UCI – Default of Credit Card Clients
Size: ~30,000 applicants
Target: Binary default indicator
Default Rate: ~22%
The dataset contains demographic, financial, and repayment history features suitable for classical ML models.
Modeling & Evaluation
Models Evaluated
Logistic Regression (baseline)
Random Forest
Histogram Gradient Boosting (winner)
Metrics Used
ROC-AUC
PR-AUC
Log Loss
Brier Score
KS Statistic
Expected Calibration Error (ECE)
Calibration
The final model is calibrated using Isotonic Regression:
Base model trained on training set
Calibration learned on validation set
Final evaluation performed on test set
This ensures reliable PD estimates suitable for decisioning.
Decision Engine (Business Logic)
Predicted PDs are converted into actions using cost-sensitive optimization:
Action	Description
APPROVE	Low predicted risk
REVIEW	Medium risk → manual underwriting
REJECT	High predicted risk
Cost Assumptions
Approve + default → High loss
Reject + good customer → Opportunity cost
Review → Operational cost
Thresholds are selected by minimizing total expected cost on the validation set and reported on the test set.
API (FastAPI)
Run the API
python -m uvicorn api.app:app --reload
Endpoints
GET /health → service health check
POST /predict → predict PD & decision
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
  },
  "missing_features": [...],
  "extra_features": []
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
API-first mindset
Realistic credit risk assumptions
Future Improvements
Add review capacity constraints
Batch prediction endpoint
SHAP-based explanations
Database logging for monitoring
CI/CD for model retraining
Author
Meera AlNeyadi
Computer Science & Software Engineering
Machine Learning • Data • Systems
