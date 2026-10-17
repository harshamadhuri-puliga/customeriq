"""Loading and cleaning the customer dataset."""
from pathlib import Path

import pandas as pd

from .config import (CATEGORICAL_FEATURES, FEATURES, ID_COL, NUMERIC_FEATURES,
                     REAL_DATA_PATH, SYNTHETIC_DATA_PATH, TARGET)
from .synth import write_synthetic


REAL_DATA_CANDIDATES = [REAL_DATA_PATH, REAL_DATA_PATH.with_name("telco.csv")]

# Column names in the newer IBM "Telco customer churn (11.1.3+)" file -> this project's names.
# Columns such as Churn Score, Churn Reason, Churn Category, Customer Status, Satisfaction Score
# and CLTV are deliberately NOT used: they are recorded after/because of churn (target leakage).
EXTENDED_RENAMES = {
    "Customer ID": "customerID", "Gender": "gender", "Married": "Partner",
    "Tenure in Months": "tenure", "Phone Service": "PhoneService",
    "Device Protection Plan": "DeviceProtection", "Premium Tech Support": "TechSupport",
    "Online Security": "OnlineSecurity", "Online Backup": "OnlineBackup",
    "Streaming TV": "StreamingTV", "Streaming Movies": "StreamingMovies",
    "Multiple Lines": "MultipleLines", "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod", "Monthly Charge": "MonthlyCharges",
    "Total Charges": "TotalCharges", "Churn Label": "Churn",
}
_TITLE_FIX = {"Month-to-Month": "Month-to-month", "One Year": "One year", "Two Year": "Two year",
              "Fiber Optic": "Fiber optic"}


def adapt_extended_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the newer IBM Telco file (50 columns) into the standard 21-column layout."""
    out = df.rename(columns=EXTENDED_RENAMES).copy()
    out["SeniorCitizen"] = (df["Senior Citizen"].astype(str).str.strip() == "Yes").astype(int)
    has_internet = df["Internet Service"].astype(str).str.strip() == "Yes"
    out["InternetService"] = (df["Internet Type"].astype(str).str.strip()
                              .where(has_internet, "No").replace({"nan": "No", "None": "No"}))
    out["InternetService"] = out["InternetService"].replace(_TITLE_FIX)
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]:
        out[col] = out[col].astype(str).str.strip().where(has_internet, "No internet service")
    no_phone = out["PhoneService"].astype(str).str.strip() == "No"
    out["MultipleLines"] = out["MultipleLines"].astype(str).str.strip().where(~no_phone, "No phone service")
    out["Contract"] = out["Contract"].astype(str).str.strip().replace(_TITLE_FIX)
    return out


def resolve_data_path(path=None):
    """Return (path, is_synthetic). Prefers the real IBM Telco CSV; falls back to synthetic data."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p, p.name == SYNTHETIC_DATA_PATH.name
    for candidate in REAL_DATA_CANDIDATES:
        if candidate.exists():
            return candidate, False
    if not SYNTHETIC_DATA_PATH.exists():
        write_synthetic()
    return SYNTHETIC_DATA_PATH, True


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues. Does NOT fit anything, so it cannot leak test information."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Customer ID" in df.columns and "Churn Label" in df.columns:
        df = adapt_extended_schema(df)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # TotalCharges is text in the raw file; blanks belong to customers with tenure 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=ID_COL)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        raise ValueError("Churn column must contain only 'Yes'/'No'.")
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load_dataset(path=None):
    """Load + clean. Returns (dataframe, source_path, is_synthetic)."""
    p, is_synth = resolve_data_path(path)
    return clean(pd.read_csv(p)), p, is_synth
