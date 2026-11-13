"""Data loading and preprocessing for the German Credit dataset (UCI / OpenML 'credit-g').

`load_german_credit` downloads the real data via OpenML.
`make_synthetic` builds a dataset with the same schema for offline smoke tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC = [
    "duration",
    "credit_amount",
    "age",
    "installment_commitment",
    "residence_since",
    "existing_credits",
    "num_dependents",
]
# Categorical columns one-hot encoded (drop_first) for the linear models.
CATEGORICAL = [
    "checking_status",
    "credit_history",
    "savings_status",
    "employment",
    "housing",
    "property_magnitude",
    "other_payment_plans",
]
GROUP_COL = "purpose"  # grouping variable for the hierarchical model


@dataclass
class CreditData:
    X_lin: pd.DataFrame  # standardised numeric + dummies (linear models); 1st column is duration_z
    X_tree: pd.DataFrame  # all features one-hot (XGBoost / SHAP)
    group: np.ndarray  # integer group index per row
    group_names: list[str]
    y: np.ndarray  # 1 = default (bad credit)


def _preprocess(raw: pd.DataFrame) -> CreditData:
    df = raw.copy()
    y = (df["class"].astype(str) == "bad").astype(int).to_numpy()

    num = df[NUMERIC].astype(float).copy()
    num["credit_amount"] = np.log(num["credit_amount"])  # heavy right tail
    num = num.rename(columns={"credit_amount": "log_credit_amount"})
    num_z = (num - num.mean()) / num.std(ddof=0)
    num_z.columns = [f"{c}_z" for c in num_z.columns]

    cats = pd.get_dummies(df[CATEGORICAL].astype(str), drop_first=True, dtype=float)
    cats.columns = [c.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("=", "eq") for c in cats.columns]
    X_lin = pd.concat([num_z, cats], axis=1)

    purpose_dum = pd.get_dummies(df[[GROUP_COL]].astype(str), dtype=float)
    X_tree = pd.concat([num, pd.get_dummies(df[CATEGORICAL].astype(str), dtype=float), purpose_dum], axis=1)
    X_tree.columns = [c.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("=", "eq").replace("[", "(").replace("]", ")") for c in X_tree.columns]

    codes, names = pd.factorize(df[GROUP_COL].astype(str), sort=True)
    return CreditData(X_lin, X_tree, codes.astype(int), [str(n) for n in names], y)


def load_german_credit(local_path: str = "data/credit.csv") -> CreditData:
    """Load German Credit from OpenML; fall back to a local CSV if OpenML is unreachable."""
    from pathlib import Path

    try:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml("credit-g", version=1, as_frame=True).frame
    except Exception as e:  # network error, OpenML down, etc.
        p = Path(local_path)
        if not p.exists():
            raise RuntimeError(
                f"OpenML download failed ({e}) and {local_path} was not found. See data/README.md."
            ) from e
        print(f"      OpenML unreachable; using {local_path}")
        raw = pd.read_csv(p)
    return _preprocess(raw)


def make_synthetic(n: int = 1000, seed: int = 0) -> CreditData:
    """Synthetic data with the German Credit schema, for offline testing only."""
    rng = np.random.default_rng(seed)
    cat_levels = {
        "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
        "credit_history": ["critical", "existing paid", "delayed previously", "all paid", "no credits/all paid"],
        "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
        "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
        "housing": ["own", "rent", "for free"],
        "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
        "other_payment_plans": ["none", "bank", "stores"],
        "purpose": ["new car", "used car", "furniture/equipment", "radio/tv", "education", "business", "repairs", "other"],
    }
    df = pd.DataFrame({k: rng.choice(v, n) for k, v in cat_levels.items()})
    df["duration"] = rng.integers(4, 72, n)
    df["credit_amount"] = np.exp(rng.normal(7.8, 0.8, n)).round()
    df["age"] = rng.integers(19, 75, n)
    df["installment_commitment"] = rng.integers(1, 5, n)
    df["residence_since"] = rng.integers(1, 5, n)
    df["existing_credits"] = rng.integers(1, 4, n)
    df["num_dependents"] = rng.integers(1, 3, n)

    purpose_eff = dict(zip(cat_levels["purpose"], rng.normal(0, 0.5, 8)))
    logit = (
        -1.2
        + 0.025 * (df["duration"] - 20)
        + 0.4 * np.log(df["credit_amount"] / 2500)
        - 0.012 * (df["age"] - 35)
        + 0.9 * (df["checking_status"] == "<0")
        - 0.6 * (df["checking_status"] == "no checking")
        + 0.5 * (df["credit_history"] == "delayed previously")
        + df["purpose"].map(purpose_eff)
    )
    df["class"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "bad", "good")
    return _preprocess(df)
