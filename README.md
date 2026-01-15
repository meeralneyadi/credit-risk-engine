# Credit Risk Engine
Production-Grade Credit Risk Modeling and Decision Engine

---

## Overview
This project implements a **production-grade credit risk system** that estimates the **Probability of Default (PD)** for credit applicants and converts those estimates into **business decisions**.

Rather than focusing only on predictive accuracy, the system emphasizes:
- probability calibration
- cost-sensitive decision making
- realistic credit policy design
- deployability via an API

The final output is a deployable decision engine capable of supporting real-world credit approval workflows.

---

## Project Objectives
The main objectives of this project are to:
- Train and evaluate multiple credit risk models
- Select a model based on both discrimination and calibration performance
- Produce reliable probability estimates suitable for decisioning
- Translate predictions into business actions (Approve / Review / Reject)
- Optimize decisions using cost-based thresholds
- Expose the system through a FastAPI service
- Validate the system using real test-set inputs

---

## Dataset
- **Source:** UCI – Default of Credit Card Clients
- **Size:** ~30,000 applicants
- **Target:** Binary default indicator
- **Default rate:** ~22%

The dataset includes demographic attributes, credit limits, payment history, bill amounts, and prior repayment behavior.

Raw and processed datasets are intentionally excluded from version control to reflect real-world data governance practices.

---

## Methodology

### Data Preparation
- Standardized feature names
- Stratified train / validation / test split
- Numerical feature scaling
- Strict prevention of data leakage between splits

---

### Model Training and Selection
The following models were trained and evaluated:
- Logistic Regression (baseline)
- Random Forest
- Histogram Gradient Boosting

Models were compared using:
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score
- KS Statistic
- Expected Calibration Error (ECE)

The **Histogram Gradient Boosting** model achieved the best balance between predictive power and calibration.

---

### Probability Calibration
The selected model was calibrated using **Isotonic Regression** to improve the reliability of predicted probabilities.

Calibration ensures that predicted PD values can be meaningfully interpreted and used in downstream business rules.

---

## Decision Engine
Predicted probabilities are converted into business actions using cost-sensitive thresholds.

### Decisions
| Decision | Description |
|--------|-------------|
| APPROVE | Low-risk applicants |
| REVIEW  | Medium-risk applicants requiring manual assessment |
| REJECT  | High-risk applicants |

### Optimized Thresholds
- Approve if PD < 0.05  
- Review if 0.05 ≤ PD < 0.67  
- Reject if PD ≥ 0.67  

Thresholds were optimized to minimize total expected cost, accounting for:
- losses from approving defaulted loans
- opportunity cost of rejecting good applicants
- operational cost of manual review

---

## Results Snapshot

### Model Performance
- ROC-AUC: ~0.77
- PR-AUC: ~0.46
- Improved Brier Score after calibration
- Stable probability estimates across risk buckets

### Policy Outcomes (Test Set)
- Approval rate: ~4–5%
- Review rate: ~90%
- Rejection rate: ~4–5%
- Default rate among approved applicants: ~4%

These results reflect a conservative credit policy focused on loss minimization and risk control.

---

## API Deployment
The system is deployed using **FastAPI** and exposes endpoints for real-time inference.

### Available Endpoints
- `GET /health` – service health check
- `POST /predict` – returns PD, decision, and applied thresholds

The API validates feature inputs and provides feedback on missing or extra features.

---

## Testing and Validation
The system was tested by sending real test-set samples through the deployed API.

Validation confirmed:
- correct PD computation
- consistent decision logic
- robust handling of missing features
- stable API responses

---

## Project Structure

    credit-risk-engine/
    ├── api/                    # FastAPI application
    ├── src/                    # Core modeling and decision logic
    ├── notebooks/              # Analysis and visualization notebooks
    ├── data/                   # Local only (ignored by Git)
    ├── artifacts/              # Models and reports (ignored by Git)
    ├── README.md
    ├── requirements.txt
    └── .gitignore

---

## Limitations and Future Work

### Limitations
- Single dataset
- Static decision thresholds
- No real-time monitoring or drift detection

### Future Improvements
- Dynamic thresholding based on review capacity
- SHAP-based explainability
- Model monitoring and retraining pipelines
- Batch inference support

---

## Author
Meera AlNeyadi  
Computer Science and Software Engineering  

