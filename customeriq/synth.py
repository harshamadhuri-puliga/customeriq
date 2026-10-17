"""Generate a synthetic dataset with the same schema as the IBM Telco churn data.

This exists ONLY so the project runs end-to-end when the real CSV is not available.
Never report metrics from synthetic data as real results.
"""
import numpy as np
import pandas as pd

from .config import SYNTHETIC_DATA_PATH


def generate(n: int = 7043, seed: int = 42, target_churn: float = 0.265) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda options, p=None: rng.choice(options, n, p=p)

    gender = pick(["Male", "Female"])
    senior = (rng.random(n) < 0.16).astype(int)
    partner = pick(["Yes", "No"], [0.48, 0.52])
    dependents = np.where(partner == "Yes", pick(["Yes", "No"], [0.40, 0.60]),
                          pick(["Yes", "No"], [0.12, 0.88]))
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])

    tenure = np.where(
        contract == "Month-to-month", rng.exponential(18, n),
        np.where(contract == "One year", rng.uniform(6, 72, n), rng.uniform(12, 72, n)),
    )
    tenure = np.clip(np.round(tenure), 1, 72).astype(int)
    tenure[rng.choice(n, 11, replace=False)] = 0  # brand-new customers (blank TotalCharges, as in real data)

    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    phone = pick(["Yes", "No"], [0.90, 0.10])
    multiple = np.where(phone == "No", "No phone service", pick(["Yes", "No"]))

    def addon(p_yes):
        return np.where(internet == "No", "No internet service", pick(["Yes", "No"], [p_yes, 1 - p_yes]))

    online_sec, online_bak = addon(0.29), addon(0.34)
    device_prot, tech_sup = addon(0.34), addon(0.29)
    stream_tv, stream_mov = addon(0.38), addon(0.39)

    paperless = pick(["Yes", "No"], [0.59, 0.41])
    payment = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                   [0.34, 0.23, 0.22, 0.21])

    monthly = np.where(internet == "No", 20.0, np.where(internet == "DSL", 45.0, 75.0))
    for col in (online_sec, online_bak, device_prot, tech_sup, stream_tv, stream_mov):
        monthly = monthly + 5.0 * (col == "Yes")
    monthly = monthly + 4.0 * (multiple == "Yes") + rng.normal(0, 3, n)
    monthly = np.round(np.clip(monthly, 18.25, 119.0), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    # Latent churn risk: realistic associations + noise so the task is not trivially separable.
    z = (
        1.3 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
        - 0.045 * tenure + 0.5 * (internet == "Fiber optic") + 0.012 * (monthly - 65)
        + 0.6 * ((tech_sup == "No") & (internet != "No")) + 0.5 * (payment == "Electronic check")
        + 0.3 * senior + 0.25 * (paperless == "Yes") - 0.3 * (dependents == "Yes")
        + rng.normal(0, 0.9, n)
    )
    lo, hi = -10.0, 10.0
    for _ in range(60):  # bisection: pick intercept so overall churn ~ target_churn
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_churn:
            hi = mid
        else:
            lo = mid
    churn = rng.random(n) < 1 / (1 + np.exp(-(z + mid)))

    ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
           for _ in range(n)]
    df = pd.DataFrame({
        "customerID": ids, "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
        "DeviceProtection": device_prot, "TechSupport": tech_sup, "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total.astype(object),
        "Churn": np.where(churn, "Yes", "No"),
    })
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "   # same quirk as the real file
    return df


def write_synthetic(path=SYNTHETIC_DATA_PATH, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(**kwargs).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print("Wrote", write_synthetic())
