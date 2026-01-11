# 💲Credit Risk Engine
Production-Grade Credit Risk Modeling and Decision Engine

---

## Overview
This project implements a production-style credit risk engine that predicts the Probability of Default (PD) for credit applicants and converts it into business decisions:

- APPROVE
- REVIEW
- REJECT

Unlike basic classification projects, this system follows real-world credit risk practices including probability calibration, cost-sensitive decision making, and deployable APIs.

---

## Key Features
- Multiple machine learning models benchmarked
- Automatic model selection based on discrimination and calibration metrics
- Isotonic calibration for reliable probability of default estimates
- Cost-based decision threshold optimization
- Strict train / validation / test separation
- FastAPI service with Swagger documentation
- Reusable core engine code in `src/`
- Policy analytics and reporting

---

## Project Structure

    credit-risk-engine/
    ├── api/
    │   ├── app.py
    │   └── schemas.py
    │
    ├── src/
    │   ├── data_prep.py
    │   ├── train.py
    │   ├── calibrate.py
    │   ├── policy.py
    │   ├── metrics.py
    │   └── config.py
    │
    ├── notebooks/
    │   ├── 01_eda_application.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_training_leaderboard.ipynb
    │   └── 04_decision_engine_demo.ipynb
    │
    ├── data/        (ignored by Git)
    ├── artifacts/   (ignored by Git)
    │
    ├── README.md
    ├── requirements.txt
    └── .gitignore

---

## Dataset
- UCI – Default of Credit Card Clients
- Approximately 30,000 applicants
- Binary default target
- Default rate around 22%

Raw and processed data are excluded from version control.

---

## Modeling and Evaluation
Models evaluated:
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting (selected)

Metrics:
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score
- KS Statistic
- Expected Calibration Error

The final model is calibrated using isotonic regression to ensure reliable probability estimates.

---

## Decision Engine
Predicted probabilities are mapped to actions using cost-sensitive optimization.

| Decision | Description |
|---------|-------------|
| APPROVE | Low risk |
| REVIEW  | Medium risk |
| REJECT  | High risk |

Thresholds minimize total expected cost on the validation set.

---

## API
The system is deployed using FastAPI and exposes endpoints for health checks and prediction.

- GET `/health`
- POST `/predict`

---

## Design Principles
- No data leakage
- Separation of modeling and business logic
- Reproducible artifacts
- API-first design
- Realistic credit risk assumptions

---

## Author
Meera AlNeyadi  
Computer Science and Software Engineering  
Engineering | Intelligence | Scale
