"""Load the trained model and score customers (used by both the API and the Streamlit app)."""
import json

import joblib
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, MODEL_DIR, NUMERIC_FEATURES
from .explain import explain_customer, retention_suggestions


class ModelNotTrainedError(RuntimeError):
    pass



def risk_level(prob: float, bands: dict) -> str:
    if prob >= bands["high_from"]:
        return "High"
    if prob >= bands["medium_from"]:
        return "Medium"
    return "Low"


def validate_customer(customer: dict, schema: dict) -> dict:
    """Return a cleaned copy of the input or raise ValueError with a readable message."""
    errors, clean = [], {}
    for f in NUMERIC_FEATURES:
        try:
            value = float(customer[f])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{f} must be a number")
            continue
        if value < 0:
            errors.append(f"{f} cannot be negative")
        clean[f] = value
    for f in CATEGORICAL_FEATURES:
        value = customer.get(f)
        if value is None:
            errors.append(f"{f} is required")
        elif value not in schema["categorical"][f]["choices"]:
            errors.append(f"{f} must be one of {schema['categorical'][f]['choices']}")
        else:
            clean[f] = value
    if errors:
        raise ValueError("; ".join(errors))
    return clean


class ModelBundle:
    def __init__(self, model_dir=MODEL_DIR):
        model_path, meta_path = model_dir / "model.joblib", model_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise ModelNotTrainedError(
                "No trained model found. Run:  python -m customeriq.train")
        self.pipeline = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text())
        self.schema = self.meta["schema"]
        self.bands = self.meta["risk_bands"]

    def predict_one(self, customer: dict, explain: bool = True) -> dict:
        clean = validate_customer(customer, self.schema)
        prob = float(self.pipeline.predict_proba(pd.DataFrame([clean])[FEATURES])[:, 1][0])
        level = risk_level(prob, self.bands)
        result = {"churn_probability": round(prob, 4), "risk_level": level,
                  "threshold": self.meta["threshold"]}
        if explain:
            result["top_factors"] = explain_customer(self.pipeline, clean, self.schema)
            result["suggestions"] = retention_suggestions(clean, level)
        return result

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score many customers at once; adds churn_probability and risk_level columns."""
        out = df.copy()
        out["churn_probability"] = self.pipeline.predict_proba(out[FEATURES])[:, 1].round(4)
        out["risk_level"] = out["churn_probability"].map(lambda p: risk_level(p, self.bands))
        return out
