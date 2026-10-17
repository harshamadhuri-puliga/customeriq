"""Run with:  pytest -q"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from customeriq.config import FEATURES, TARGET
from customeriq.data import clean
from customeriq.db import QUERIES, load_scored_customers, run_query
from customeriq.predict import ModelBundle, validate_customer
from customeriq.synth import generate
from customeriq.train import choose_threshold, train


def _trained(tmp: Path):
    data_path = tmp / "data.csv"
    generate(n=1500, seed=1).to_csv(data_path, index=False)
    model_dir = tmp / "models"
    train(data_path, model_dir=model_dir, verbose=False)
    return data_path, model_dir


def test_clean_fixes_total_charges_and_target():
    df = clean(generate(n=500, seed=2))
    assert df["TotalCharges"].notna().all()
    assert set(df[TARGET].unique()) <= {0, 1}
    assert all(c in df.columns for c in FEATURES)


def test_threshold_prefers_catching_churners_when_misses_cost_more():
    y = np.array([1] * 20 + [0] * 80)
    proba = np.concatenate([np.linspace(0.3, 0.9, 20), np.linspace(0.0, 0.6, 80)])
    low_cost_miss, _ = choose_threshold(y, proba, cost_fn=1, cost_fp=1)
    high_cost_miss, _ = choose_threshold(y, proba, cost_fn=10, cost_fp=1)
    assert high_cost_miss <= low_cost_miss


def test_train_predict_explain_and_sql():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        data_path, model_dir = _trained(tmp)
        bundle = ModelBundle(model_dir)
        assert bundle.meta["test_metrics"]["roc_auc"] > 0.7

        risky = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
                 "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                 "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                 "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                 "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                 "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0, "TotalCharges": 180.0}
        safe = {**risky, "tenure": 70, "Contract": "Two year", "InternetService": "DSL",
                "TechSupport": "Yes", "OnlineSecurity": "Yes", "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 55.0, "TotalCharges": 3850.0}
        r, s = bundle.predict_one(risky), bundle.predict_one(safe)
        assert 0 <= s["churn_probability"] < r["churn_probability"] <= 1
        assert r["risk_level"] in {"Medium", "High"} and s["risk_level"] == "Low"
        assert r["top_factors"] and r["suggestions"]

        db = tmp / "test.db"
        n = load_scored_customers(db, bundle=bundle, data_path=data_path)
        assert n > 1000
        for sql in QUERIES.values():
            assert len(run_query(sql, db)) >= 1


def test_validation_rejects_bad_input():
    with tempfile.TemporaryDirectory() as d:
        _, model_dir = _trained(Path(d))
        schema = ModelBundle(model_dir).schema
    bad = {f: "x" for f in FEATURES}
    bad["tenure"] = -3
    try:
        validate_customer(bad, schema)
    except ValueError as err:
        assert "tenure" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_extended_schema_is_adapted_and_leaky_columns_dropped():
    raw = pd.DataFrame({
        "Customer ID": ["a", "b"], "Gender": ["Male", "Female"], "Senior Citizen": ["No", "Yes"],
        "Married": ["Yes", "No"], "Dependents": ["No", "No"], "Tenure in Months": [5, 40],
        "Phone Service": ["No", "Yes"], "Multiple Lines": ["No", "Yes"],
        "Internet Service": ["No", "Yes"], "Internet Type": [None, "Fiber Optic"],
        "Online Security": ["No", "Yes"], "Online Backup": ["No", "No"],
        "Device Protection Plan": ["No", "No"], "Premium Tech Support": ["No", "Yes"],
        "Streaming TV": ["No", "Yes"], "Streaming Movies": ["No", "No"],
        "Contract": ["Month-to-Month", "Two Year"], "Paperless Billing": ["Yes", "No"],
        "Payment Method": ["Credit Card", "Mailed Check"], "Monthly Charge": [20.0, 90.0],
        "Total Charges": [100.0, 3600.0], "Churn Label": ["Yes", "No"],
        "Churn Score": [90, 10], "Churn Reason": ["Price", None], "Satisfaction Score": [1, 5]})
    df = clean(raw)
    assert df.loc[0, "InternetService"] == "No" and df.loc[0, "OnlineSecurity"] == "No internet service"
    assert df.loc[0, "MultipleLines"] == "No phone service"
    assert df.loc[1, "InternetService"] == "Fiber optic" and df.loc[1, "Contract"] == "Two year"
    assert df.loc[0, "SeniorCitizen"] == 0 and df.loc[1, "SeniorCitizen"] == 1
    assert df[TARGET].tolist() == [1, 0]
