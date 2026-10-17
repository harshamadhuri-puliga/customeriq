"""Per-customer explanations and retention suggestions.

Explanations use a simple, dependency-free "swap to typical value" method: for each feature we
replace the customer's value with the typical value (median / most common) and see how much the
predicted churn probability drops. A positive number means that feature is associated with HIGHER
predicted risk for this customer. This shows model associations, not proven causes.
"""
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def baseline_row(schema: dict) -> dict:
    base = {c: schema["numeric"][c]["median"] for c in NUMERIC_FEATURES}
    base.update({c: schema["categorical"][c]["default"] for c in CATEGORICAL_FEATURES})
    return base


def explain_customer(pipeline, customer: dict, schema: dict, top_k: int = 5):
    """Return the top features pushing this customer's churn risk up (or down)."""
    row = pd.DataFrame([{f: customer[f] for f in FEATURES}])
    p_full = float(pipeline.predict_proba(row)[:, 1][0])
    base = baseline_row(schema)

    variants = []
    for f in FEATURES:
        alt = row.copy()
        alt[f] = base[f]
        variants.append(alt)
    p_alt = pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]

    effects = [{"feature": f, "value": customer[f], "effect": round(p_full - float(p), 4)}
               for f, p in zip(FEATURES, p_alt) if abs(p_full - float(p)) > 1e-4]
    effects.sort(key=lambda e: abs(e["effect"]), reverse=True)
    return effects[:top_k]


def retention_suggestions(customer: dict, risk_level: str):
    """Rule-based decision support. These are ideas to review, not guaranteed fixes."""
    if risk_level == "Low":
        return ["No action needed. Keep monitoring and maintain normal service quality."]
    tips = []
    if customer.get("Contract") == "Month-to-month":
        tips.append("Offer a discounted 1-year or 2-year contract; month-to-month customers churn most.")
    if customer.get("InternetService") != "No" and customer.get("TechSupport") == "No":
        tips.append("Offer a free trial of Tech Support; customers without it are more likely to leave.")
    if customer.get("InternetService") != "No" and customer.get("OnlineSecurity") == "No":
        tips.append("Bundle Online Security to raise the value of the plan.")
    if customer.get("InternetService") == "Fiber optic" and customer.get("MonthlyCharges", 0) > 80:
        tips.append("Review the plan: a high Fiber bill may justify a loyalty discount or cheaper tier.")
    if customer.get("PaymentMethod") in ("Electronic check", "Mailed check", "Mailed Check"):
        tips.append("Encourage automatic payment (bank/credit card) with a small incentive.")
    if customer.get("tenure", 99) < 6:
        tips.append("New customer: schedule an onboarding check-in call in the first 90 days.")
    if not tips:
        tips.append("Contact the customer to review satisfaction and any open service complaints.")
    return tips + ["These suggestions are decision support only; test them before assuming they reduce churn."]
