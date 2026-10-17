"""Central configuration: paths, column definitions, business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REAL_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"      # IBM Telco dataset (you download this)
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_telco.csv"       # generated fallback for testing
DB_PATH = ROOT / "customeriq.db"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Business assumption used to choose the decision threshold (editable via CLI flags).
# Missing a customer who leaves is assumed 3x as costly as wasting one retention offer.
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0

RANDOM_STATE = 42
