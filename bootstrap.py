"""Bootstrap confidence intervals for test metrics and paired metric differences.

Usage: python -m src.bootstrap --out results
Resamples the held-out test applicants (predictions are fixed; models are not refit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E

MODELS = {"Logistic regression (MLE)": "p_lr", "Bayesian LR": "p_bayes", "Hierarchical Bayes": "p_hier", "XGBoost": "p_xgb"}
METRICS = ["accuracy", "auc", "brier", "log_loss", "ece"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    d = pd.read_csv(out / "test_predictions.csv")
    y = d["y"].to_numpy()
    rng = np.random.default_rng(a.seed)
    n = len(y)
    boots = {m: {k: [] for k in METRICS} for m in MODELS}
    for _ in range(a.B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for m, col in MODELS.items():
            r = E.metrics_row(m, y[i], d[col].to_numpy()[i])
            for k in METRICS:
                boots[m][k].append(r[k])
    rows = []
    for m, col in MODELS.items():
        pt = E.metrics_row(m, y, d[col].to_numpy())
        row = {"model": m}
        for k in METRICS:
            lo, hi = np.quantile(boots[m][k], [0.025, 0.975])
            row[k] = f"{pt[k]:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "metrics_bootstrap_ci.csv", index=False)

    base = "Logistic regression (MLE)"
    diffs = {}
    for m in MODELS:
        if m == base:
            continue
        for k in ["auc", "brier", "log_loss", "ece"]:
            dv = np.array(boots[m][k]) - np.array(boots[base][k])
            lo, hi = np.quantile(dv, [0.025, 0.975])
            diffs[f"{m} minus {base} | {k}"] = {"mean_diff": float(dv.mean()), "ci95": [float(lo), float(hi)],
                                              "excludes_zero": bool(lo > 0 or hi < 0)}
    (out / "paired_differences.json").write_text(json.dumps(diffs, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    for k, v in diffs.items():
        print(k, round(v["mean_diff"], 4), [round(x, 4) for x in v["ci95"]], "SIG" if v["excludes_zero"] else "")


if __name__ == "__main__":
    main()
