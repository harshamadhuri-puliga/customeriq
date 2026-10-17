-- MySQL version of the customers table (SQLite is created automatically by customeriq.db).
CREATE DATABASE IF NOT EXISTS customeriq;
USE customeriq;

CREATE TABLE IF NOT EXISTS customers (
    customerID        VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10),
    SeniorCitizen     TINYINT,
    Partner           VARCHAR(5),
    Dependents        VARCHAR(5),
    tenure            INT,
    PhoneService      VARCHAR(5),
    MultipleLines     VARCHAR(20),
    InternetService   VARCHAR(20),
    OnlineSecurity    VARCHAR(25),
    OnlineBackup      VARCHAR(25),
    DeviceProtection  VARCHAR(25),
    TechSupport       VARCHAR(25),
    StreamingTV       VARCHAR(25),
    StreamingMovies   VARCHAR(25),
    Contract          VARCHAR(20),
    PaperlessBilling  VARCHAR(5),
    PaymentMethod     VARCHAR(30),
    MonthlyCharges    DECIMAL(8,2),
    TotalCharges      DECIMAL(10,2),
    Churn             TINYINT,
    churn_probability DECIMAL(5,4),
    risk_level        VARCHAR(6)
);

-- Example analysis queries (same as customeriq/db.py):
SELECT Contract, COUNT(*) AS customers, ROUND(100 * AVG(Churn), 1) AS churn_pct
FROM customers GROUP BY Contract ORDER BY churn_pct DESC;

SELECT COUNT(*) AS high_risk_customers, ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk
FROM customers WHERE risk_level = 'High';
