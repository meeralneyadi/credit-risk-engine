from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="feature_name -> value")


class PredictResponse(BaseModel):
    pd: float
    decision: str
    thresholds: Dict[str, float]
    missing_features: List[str]
    extra_features: List[str]
