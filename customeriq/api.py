"""REST API for CustomerIQ.   Run:  uvicorn customeriq.api:app --reload   (docs at /docs)"""
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import ModelBundle, ModelNotTrainedError

class Customer(BaseModel):
    """Categorical values are validated against the categories the model was trained on.
    A wrong value returns 422 listing the allowed choices (see GET /model-info)."""
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 4,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed Check",
        "MonthlyCharges": 85.5, "TotalCharges": 342.0}}}


class BatchRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=1000)


app = FastAPI(title="CustomerIQ API", version="1.0.0",
              description="Predict customer churn risk. Probabilities are estimates, not guarantees.")


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle()


def bundle_or_503() -> ModelBundle:
    try:
        return get_bundle()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    try:
        get_bundle()
        return {"status": "ok", "model_loaded": True}
    except ModelNotTrainedError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model-info")
def model_info():
    meta = bundle_or_503().meta
    keys = ("version", "trained_at", "data_source", "synthetic_data", "best_model", "threshold",
            "risk_bands", "test_metrics", "comparison")
    return {k: meta[k] for k in keys}


@app.post("/predict")
def predict(customer: Customer):
    try:
        return bundle_or_503().predict_one(customer.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    bundle = bundle_or_503()
    try:
        return {"results": [bundle.predict_one(c.model_dump(), explain=False) for c in req.customers]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
