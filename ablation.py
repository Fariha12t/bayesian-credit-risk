"""Ablation for research question 2: does *partial pooling* help, or is it just the extra purpose information?

Fits a flat (no-pooling) Bayesian LR that receives loan-purpose dummies, on the same split and seed as run_all,
and compares it with the hierarchical model's saved test predictions.
Usage: python -m src.ablation --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import data as D
from . import evaluate as E
from . import models as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    cd = D.load_german_credit()
    tr, te = train_test_split(np.arange(len(cd.y)), test_size=0.3, stratify=cd.y, random_state=a.seed)
    P = np.eye(len(cd.group_names))[cd.group][:, 1:]  # purpose dummies, first level dropped
    X = np.hstack([cd.X_lin.to_numpy(), P])
    names = list(cd.X_lin.columns) + [f"purpose_{g}" for g in cd.group_names[1:]]
    idata = M.fit_bayes_lr(X[tr], cd.y[tr], names, seed=a.seed)
    p_flat = E.summarise_draws(M.predict_bayes_lr(idata, X[te]))["mean"]
    d = pd.read_csv(out / "test_predictions.csv")
    d["p_flat_purpose"] = p_flat
    d.to_csv(out / "test_predictions.csv", index=False)
    y = d["y"].to_numpy()
    rng = np.random.default_rng(a.seed)
    rows, diffs = {}, {k: [] for k in ["auc", "brier", "log_loss"]}
    for _ in range(a.B):
        i = rng.integers(0, len(y), len(y))
        h = E.metrics_row("h", y[i], d["p_hier"].to_numpy()[i])
        f = E.metrics_row("f", y[i], d["p_flat_purpose"].to_numpy()[i])
        for k in diffs:
            diffs[k].append(h[k] - f[k])
    res = {"flat_purpose_metrics": E.metrics_row("Bayesian LR + purpose dummies (no pooling)", y, p_flat),
           "hier_minus_flat": {k: {"mean": float(np.mean(v)), "ci95": [float(x) for x in np.quantile(v, [0.025, 0.975])]}
                                for k, v in diffs.items()}}
    (out / "ablation_pooling.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
