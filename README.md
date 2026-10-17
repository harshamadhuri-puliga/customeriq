# CustomerIQ: ML-Powered Customer Retention Platform

Predicts which telecom customers are likely to churn, explains why, and suggests retention actions.
Includes a trained model, REST API (FastAPI), 4-page dashboard (Streamlit), SQL analysis and tests.

## Quick start
```bash
pip install -r requirements.txt

# 1. Get the real data (recommended): IBM Telco Customer Churn
#    https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
#    Save it as data/Telco-Customer-Churn.csv (or data/telco.csv). Both the classic 21-column file
#    and the newer 50-column IBM file are supported; leaky columns (Churn Score, Churn Reason,
#    Satisfaction Score, CLTV...) are ignored automatically.
#    Without any file, a SYNTHETIC demo dataset is generated.

python -m customeriq.train          # trains 3 models, tunes threshold, saves models/
python -m customeriq.db load        # loads scored customers into SQLite + runs SQL analysis
python -m streamlit run app.py      # dashboard  -> http://localhost:8501
uvicorn customeriq.api:app --reload # API docs   -> http://localhost:8000/docs
pytest -q                           # tests
```

## How it works
| Step | File | What happens |
|---|---|---|
| Clean | `customeriq/data.py` | Fixes blank `TotalCharges`, strips text, removes duplicates, encodes `Churn` |
| Train | `customeriq/train.py` | 60/20/20 stratified split; Logistic Regression vs Random Forest vs Gradient Boosting; preprocessing inside a `Pipeline` (no leakage) |
| Threshold | `customeriq/train.py` | Cut-off chosen on **validation** data by minimising business cost (missed churner = 3x a wasted offer; change with `--cost-fn/--cost-fp`) |
| Evaluate | `customeriq/train.py` | Test set used once: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix vs "everyone stays" baseline |
| Explain | `customeriq/explain.py` | For each customer, which features raise or lower predicted risk (associations, not causes) |
| Serve | `customeriq/api.py`, `app.py` | `/predict`, `/predict/batch`, `/model-info`; dashboard pages Overview, Predict Churn, Model Insights, At-Risk Customers |
| SQL | `customeriq/db.py`, `sql/` | SQLite out of the box; `sql/mysql_schema.sql` for MySQL |

## Docker
```bash
docker build -t customeriq .
docker run -p 8000:8000 customeriq
```

## Before you put this on your resume
1. Train on the **real** IBM dataset. Metrics from the synthetic file are meaningless.
2. Copy your real numbers from `models/metadata.json` (never invent them).
3. Deploy it (Render, Hugging Face Spaces, Streamlit Community Cloud) and add the link.
4. Add screenshots and a short demo video to this README.

Suggested bullets (fill in real values):
- Built an end-to-end churn prediction system on 7,043 telecom customers using Python, Pandas and scikit-learn; compared Logistic Regression, Random Forest and Gradient Boosting using ROC-AUC, precision and recall (best ROC-AUC: **__**).
- Tuned the decision threshold against a business cost assumption, reaching **__** recall at **__** precision on a held-out test set.
- Served the model through a FastAPI REST API and a Streamlit dashboard with per-customer explanations and retention suggestions; containerised with Docker and covered by pytest tests.

## Notes
- The dashboard's Overview and At-Risk pages score ALL customers, including ones the model trained on,
  so those risk counts are slightly optimistic. Only the Model Insights metrics (held-out test set) are
  honest performance numbers.

## Limitations
Predictions are estimates from historical patterns. Feature effects show association, not causation. The
dataset is a single public telecom sample; performance on another company's customers is unknown.
