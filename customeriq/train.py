"""Train, compare, threshold-tune and save the churn model.

Leakage safeguards:
  * stratified train / validation / test split (60/20/20)
  * all preprocessing lives inside the sklearn Pipeline, so it is fit on training rows only
  * the model is picked and the decision threshold is tuned on VALIDATION data
  * TEST data is touched once, at the end, for the final report

Run:  python -m customeriq.train [--data path.csv] [--cost-fn 5] [--cost-fp 1]
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .config import (CATEGORICAL_FEATURES, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, FEATURES,
                     MODEL_DIR, NUMERIC_FEATURES, RANDOM_STATE, TARGET)
from .data import load_dataset


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models(seed: int):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                            max_iter=200, random_state=seed),
    }


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, proba, cost_fn, cost_fp):
    """Pick the probability cut-off that minimises expected business cost on validation data."""
    best_t, best_cost = 0.5, np.inf
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = proba >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def segment_report(X_test, y_test, proba, threshold, min_size=30):
    """Test-set performance for customer groups, to spot groups the model serves less well."""
    groups = {
        "Contract": X_test["Contract"].astype(str),
        "Internet service": X_test["InternetService"].astype(str),
        "Senior citizen": X_test["SeniorCitizen"].map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"}),
        "Months as customer": pd.cut(X_test["tenure"], [-1, 11, 23, 47, 100],
                                     labels=["0-11", "12-23", "24-47", "48+"]).astype(str),
    }
    rows = []
    for name, series in groups.items():
        for value in sorted(series.dropna().unique(), key=str):
            mask = (series == value).values
            n = int(mask.sum())
            if n < min_size:
                continue
            yt, pr = y_test[mask], proba[mask]
            pred = pr >= threshold
            has_churners = int(yt.sum()) > 0
            rows.append({
                "group_by": name, "group": str(value), "customers": n,
                "churn_rate": float(yt.mean()), "flagged_rate": float(pred.mean()),
                "precision": float(precision_score(yt, pred, zero_division=0)) if pred.any() else None,
                "recall": float(recall_score(yt, pred, zero_division=0)) if has_churners else None,
                "roc_auc": float(roc_auc_score(yt, pr)) if has_churners and len(set(yt)) == 2 else None,
            })
    return rows


def train(data_path=None, model_dir=MODEL_DIR, cost_fn=COST_FALSE_NEGATIVE,
          cost_fp=COST_FALSE_POSITIVE, seed=RANDOM_STATE, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    df, source, is_synth = load_dataset(data_path)
    X, y = df[FEATURES], df[TARGET].values
    log(f"Loaded {len(df):,} customers from {source.name} | churn rate {y.mean():.1%}"
        + ("  [SYNTHETIC DATA - metrics are NOT real results]" if is_synth else ""))

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp,
                                                      random_state=seed)

    # 1) compare candidates on validation data
    fitted, comparison = {}, []
    for name, estimator in candidate_models(seed).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)]).fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        fitted[name] = pipe
        comparison.append({
            "model": name,
            "val_roc_auc": float(roc_auc_score(y_val, proba)),
            "val_pr_auc": float(average_precision_score(y_val, proba)),
            "val_f1_at_0.5": float(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0)),
        })
        log(f"  {name:<20} val ROC-AUC {comparison[-1]['val_roc_auc']:.3f} | "
            f"PR-AUC {comparison[-1]['val_pr_auc']:.3f} | F1@0.5 {comparison[-1]['val_f1_at_0.5']:.3f}")

    best_name = max(comparison, key=lambda r: r["val_roc_auc"])["model"]
    best = fitted[best_name]

    # 2) tune the decision threshold on validation data
    val_proba = best.predict_proba(X_val)[:, 1]
    threshold, _ = choose_threshold(y_val, val_proba, cost_fn, cost_fp)

    # 3) final, one-time evaluation on the untouched test set
    test_proba = best.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_proba, threshold)
    test_metrics["roc_auc"] = float(roc_auc_score(y_test, test_proba))
    test_metrics["pr_auc"] = float(average_precision_score(y_test, test_proba))
    majority = max(y_test.mean(), 1 - y_test.mean())
    test_metrics["majority_class_accuracy"] = float(majority)  # the "predict everyone stays" baseline
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int)

    log(f"\nBest model: {best_name} | threshold {threshold:.2f}")
    log(f"Test  ROC-AUC {test_metrics['roc_auc']:.3f} | precision {test_metrics['precision']:.3f} | "
        f"recall {test_metrics['recall']:.3f} | F1 {test_metrics['f1']:.3f} | "
        f"accuracy {test_metrics['accuracy']:.3f} (always-'stays' baseline {majority:.3f})")

    # 4) global importance on the test set (permutation importance on raw columns)
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=seed, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std)),
        key=lambda r: r["importance"], reverse=True)

    # 5) metadata the API / UI need (input choices, baselines for explanations)
    schema = {
        "numeric": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                        "median": float(df[c].median())} for c in NUMERIC_FEATURES},
        "categorical": {c: {"choices": df[c].value_counts().index.tolist(),
                            "default": str(df[c].mode().iloc[0])} for c in CATEGORICAL_FEATURES},
    }
    metadata = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source.name,
        "synthetic_data": bool(is_synth),
        "n_customers": int(len(df)),
        "churn_rate": float(y.mean()),
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "best_model": best_name,
        "comparison": comparison,
        "threshold": threshold,
        "risk_bands": {"medium_from": round(threshold / 2, 3), "high_from": threshold},
        "cost_assumptions": {"false_negative": cost_fn, "false_positive": cost_fp},
        "test_metrics": test_metrics,
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "feature_importance": importance,
        "segments": segment_report(X_test, y_test, test_proba, threshold),
        "schema": schema,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log(f"Saved model + metadata to {model_dir}")
    return metadata


def main():
    ap = argparse.ArgumentParser(description="Train the CustomerIQ churn model")
    ap.add_argument("--data", help="path to a Telco-style CSV (default: real IBM CSV, else synthetic)")
    ap.add_argument("--cost-fn", type=float, default=COST_FALSE_NEGATIVE,
                    help="cost of missing a customer who churns")
    ap.add_argument("--cost-fp", type=float, default=COST_FALSE_POSITIVE,
                    help="cost of a retention offer sent to a customer who would have stayed")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    a = ap.parse_args()
    train(a.data, cost_fn=a.cost_fn, cost_fp=a.cost_fp, seed=a.seed)


if __name__ == "__main__":
    main()
