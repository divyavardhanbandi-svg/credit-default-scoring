"""
Credit Default Scoring
======================
End-to-end project: load data -> preprocess -> train -> evaluate -> score -> save.

Usage
-----
    pip install pandas numpy scikit-learn matplotlib joblib

    # Option 1: run on built-in synthetic data (no download needed)
    python credit_default_scoring.py

    # Option 2: run on a real CSV (e.g. UCI "Default of Credit Card Clients")
    python credit_default_scoring.py --data UCI_Credit_Card.csv --target default.payment.next.month

Outputs (in ./outputs)
----------------------
    best_model.joblib     trained pipeline (preprocessing + model)
    roc_curve.png         ROC curves for both models
    scored_test_set.csv   test set with default probability and credit score
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def make_synthetic_data(n=8000, seed=RANDOM_STATE):
    """Generate a realistic-looking credit dataset with ~20% default rate."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    income = rng.lognormal(mean=10.8, sigma=0.5, size=n)
    credit_limit = np.clip(income * rng.uniform(0.2, 0.8, n), 1000, None)
    utilization = np.clip(rng.beta(2, 3, n), 0, 1)
    late_payments = rng.poisson(0.8, n)
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40)
    loan_amount = credit_limit * utilization
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.40, 0.15])
    education = rng.choice(["HIGH_SCHOOL", "BACHELOR", "MASTER", "OTHER"], n,
                           p=[0.35, 0.40, 0.15, 0.10])

    # Hidden relationship between features and default risk
    logit = (
        -3.0
        + 2.8 * utilization
        + 0.55 * late_payments
        - 0.03 * employment_years
        - 0.000004 * income
        - 0.012 * (age - 40)
        + 0.35 * (home_ownership == "RENT")
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "credit_limit": credit_limit.round(2),
            "utilization": utilization.round(4),
            "late_payments": late_payments,
            "employment_years": employment_years.round(1),
            "loan_amount": loan_amount.round(2),
            "home_ownership": home_ownership,
            "education": education,
            "default": default,
        }
    )

    # Inject a little missing data so the imputers have work to do
    for col in ["income", "employment_years"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


def load_data(path, target):
    if path is None:
        print("No --data supplied, using synthetic data.")
        return make_synthetic_data(), "default"
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    # Drop obvious ID columns if present
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("id", "customer_id")],
                 errors="ignore")
    return df, target


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)]
    )


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------
def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic: max separation between good and bad CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(name, model, X_test, y_test, threshold=0.5):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(y_test, prob)
    gini = 2 * auc - 1
    ks = ks_statistic(y_test, prob)

    print(f"\n===== {name} =====")
    print(f"AUC : {auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"KS  : {ks:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=3))
    return {"name": name, "auc": auc, "prob": prob}


# --------------------------------------------------------------------------
# 4. Credit score conversion
# --------------------------------------------------------------------------
def probability_to_score(p_default, base_score=600, base_odds=50, pdo=20):
    """
    Convert probability of default to a scorecard-style credit score.

    base_score : score at which odds (good:bad) equal base_odds
    pdo        : points to double the odds
    Higher score = lower risk.
    """
    p = np.clip(p_default, 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.round(offset + factor * np.log(odds_good)).astype(int)


def risk_band(score):
    bins = [-np.inf, 520, 580, 640, 700, np.inf]
    labels = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
    return pd.cut(score, bins=bins, labels=labels)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, target = load_data(args.data, args.target)
    print(f"Loaded {len(df):,} rows, default rate = {df[target].mean():.2%}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        # Lower threshold for the unbalanced boosting model so recall isn't ignored
        thr = 0.5 if name == "Logistic Regression" else float(y_train.mean())
        results.append(evaluate(name, pipe, X_test, y_test, threshold=thr))

    # ROC plot
    plt.figure(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Credit Default Scoring - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # Pick best model by AUC
    best = max(results, key=lambda r: r["auc"])
    best_model = fitted[best["name"]]
    print(f"\nBest model: {best['name']} (AUC={best['auc']:.4f})")

    # Score the test set
    scored = X_test.copy()
    scored["actual_default"] = y_test.values
    scored["default_probability"] = best["prob"].round(4)
    scored["credit_score"] = probability_to_score(best["prob"])
    scored["risk_band"] = risk_band(scored["credit_score"])
    scored.to_csv(os.path.join(OUTPUT_DIR, "scored_test_set.csv"), index=False)

    print("\nDefault rate by risk band (sanity check - should fall as score rises):")
    print(
        scored.groupby("risk_band", observed=True)["actual_default"]
        .agg(["count", "mean"])
        .rename(columns={"count": "customers", "mean": "default_rate"})
    )

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.joblib"))
    print(f"\nSaved model, ROC plot and scored data to ./{OUTPUT_DIR}/")

    # Example: score a single new applicant (works with the synthetic schema)
    if args.data is None:
        applicant = pd.DataFrame(
            [{
                "age": 34, "income": 52000, "credit_limit": 15000, "utilization": 0.72,
                "late_payments": 3, "employment_years": 4.0, "loan_amount": 10800,
                "home_ownership": "RENT", "education": "BACHELOR",
            }]
        )
        p = best_model.predict_proba(applicant)[:, 1]
        print(
            f"\nSample applicant -> P(default)={p[0]:.2%}, "
            f"score={probability_to_score(p)[0]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Default Scoring")
    parser.add_argument("--data", default=None, help="Path to a CSV file (optional)")
    parser.add_argument("--target", default="default", help="Target column name")
    main(parser.parse_args())
