"""Store customers and risk scores in a SQL database and query them.

Uses SQLite out of the box (no setup). For MySQL, use sql/mysql_schema.sql and the same queries.
CLI:  python -m customeriq.db load      # load + score customers into customeriq.db
"""
import sqlite3

from .config import DB_PATH
from .data import load_dataset
from .predict import ModelBundle

QUERIES = {
    "Churn rate by contract": """
        SELECT Contract, COUNT(*) AS customers, ROUND(100.0 * AVG(Churn), 1) AS churn_pct
        FROM customers GROUP BY Contract ORDER BY churn_pct DESC""",
    "Churn rate by tenure band": """
        SELECT CASE WHEN tenure < 12 THEN '0-11 months' WHEN tenure < 24 THEN '12-23 months'
                    WHEN tenure < 48 THEN '24-47 months' ELSE '48+ months' END AS tenure_band,
               COUNT(*) AS customers, ROUND(100.0 * AVG(Churn), 1) AS churn_pct
        FROM customers GROUP BY tenure_band ORDER BY MIN(tenure)""",
    "Revenue at risk (High-risk customers)": """
        SELECT COUNT(*) AS high_risk_customers, ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk
        FROM customers WHERE risk_level = 'High'""",
}


def load_scored_customers(db_path=DB_PATH, bundle=None, data_path=None):
    bundle = bundle or ModelBundle()
    df, _, _ = load_dataset(data_path)
    scored = bundle.score_dataframe(df)
    with sqlite3.connect(db_path) as con:
        scored.to_sql("customers", con, if_exists="replace", index=False)
    return len(scored)


def run_query(sql, db_path=DB_PATH):
    import pandas as pd
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["load"]:
        print(f"Loaded {load_scored_customers():,} scored customers into {DB_PATH}")
        for title, sql in QUERIES.items():
            print(f"\n{title}\n{run_query(sql).to_string(index=False)}")
    else:
        print("usage: python -m customeriq.db load")
