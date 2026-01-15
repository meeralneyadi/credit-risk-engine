from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

ARTIFACTS = ROOT / "artifacts"
ART_MODELS = ARTIFACTS / "models"
ART_REPORTS = ARTIFACTS / "reports"
ART_METRICS = ARTIFACTS / "metrics"

RAW_XLS_NAME = "default of credit card clients.xls"

RANDOM_STATE = 42
