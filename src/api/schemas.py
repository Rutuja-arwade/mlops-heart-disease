from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXAMPLE = {"age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 286, "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
           "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7}


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})
    age: int = Field(..., ge=18, le=100)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(..., ge=70, le=250)
    chol: float = Field(..., ge=100, le=650)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(..., ge=60, le=230)
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=0, le=7)
    slope: Literal[1, 2, 3]
    ca: Optional[Literal[0, 1, 2, 3]] = None
    thal: Optional[Literal[3, 6, 7]] = None

    @field_validator("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", mode="before")
    @classmethod
    def int_codes(cls, v):
        if isinstance(v, str) and v.strip().lstrip("-").replace(".", "", 1).isdigit():
            v = float(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    risk_level: Literal["low", "moderate", "high"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    model_version: str
    count: int
    predictions: list[dict]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
