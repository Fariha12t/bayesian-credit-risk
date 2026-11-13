"""End-to-end pipeline: python -m src.run_all --data german --out results"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import data as D
from . import evaluate as E
from . import models as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["german", "synthetic"], default="german")
    ap.add_argument("--out", default="results")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed

    print(f"[1/6] Loading data: {args.data}")
    cd = D.load_german_credit() if args.data == "german" else D.make_synthetic(seed=rng_seed)
    print(f"      n={len(cd.y)}, default rate={cd.y.mean():.3f}, features={cd.X_lin.shape[1]}, groups={len(cd.group_names)}")

    idx = np.arange(len(cd.y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=cd.y, random_state=rng_seed)
    Xl_tr, Xl_te = cd.X_lin.iloc[tr].to_numpy(), cd.X_lin.iloc[te].to_numpy()
    Xt_tr, Xt_te = cd.X_tree.iloc[tr], cd.X_tree.iloc[te]
    g_tr, g_te = cd.group[tr], cd.group[te]
    y_tr, y_te = cd.y[tr], cd.y[te]
    feats = list(cd.X_lin.columns)

    print("[2/6] Fitting baselines (logistic regression, XGBoost)")
    lr = LogisticRegression(max_iter=2000).fit(Xl_tr, y_tr)
    p_lr = lr.predict_proba(Xl_te)[:, 1]
    xgbm = M.fit_xgb(Xt_tr, y_tr, seed=rng_seed)
    p_xgb = xgbm.predict_proba(Xt_te)[:, 1]

    print("[3/6] Sampling Bayesian logistic regression")
    idata_b = M.fit_bayes_lr(Xl_tr, y_tr, feats, args.draws, args.tune, args.chains, rng_seed)
    d_b = M.predict_bayes_lr(idata_b, Xl_te)

    print("[4/6] Sampling hierarchical logistic regression")
    idata_h = M.fit_hier_lr(Xl_tr, g_tr, y_tr, cd.group_names, feats, args.draws, args.tune, args.chains, rng_seed)
    d_h = M.predict_hier_lr(idata_h, Xl_te, g_te)

    print("[5/6] Evaluating")
    sb, sh = E.summarise_draws(d_b), E.summarise_draws(d_h)
    rows = [
        E.metrics_row("XGBoost", y_te, p_xgb),
        E.metrics_row("Logistic regression (MLE)", y_te, p_lr),
        E.metrics_row("Bayesian LR (posterior mean)", y_te, sb["mean"]),
        E.metrics_row("Hierarchical Bayes (posterior mean)", y_te, sh["mean"]),
    ]
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False))

    E.plot_calibration(
        y_te,
        {"XGBoost": p_xgb, "Bayesian LR": sb["mean"], "Hierarchical Bayes": sh["mean"]},
        out / "figures" / "calibration.png",
    )
    sel = E.selective_prediction(y_te, sb["mean"], sb["std"], p_xgb)
    sel.round(4).to_csv(out / "selective_prediction.csv", index=False)
    E.plot_selective(sel, out / "figures" / "selective_prediction.png")
    E.plot_interval_examples(y_te, sb, out / "figures" / "credible_intervals.png")
    unc = E.interval_coverage(y_te, sb["mean"], sb["std"], p_xgb)
    unc["mean_94pct_interval_width"] = float((sb["hi"] - sb["lo"]).mean())
    print("Uncertainty diagnostics:", unc)

    print("[6/6] Diagnostics and figures")
    summ_b = E.arviz_summary(idata_b, ["alpha", "beta"])
    summ_h = E.arviz_summary(idata_h, ["mu_a", "sigma_a", "sigma_b", "a", "b_dur", "beta"])
    summ_b.round(3).to_csv(out / "posterior_summary_bayes_lr.csv")
    summ_h.round(3).to_csv(out / "posterior_summary_hierarchical.csv")
    diag = {
        "bayes_lr": {
            "max_rhat": float(summ_b["r_hat"].max()),
            "min_ess_bulk": float(summ_b["ess_bulk"].min()),
            "divergences": E.n_divergences(idata_b),
        },
        "hierarchical": {
            "max_rhat": float(summ_h["r_hat"].max()),
            "min_ess_bulk": float(summ_h["ess_bulk"].min()),
            "divergences": E.n_divergences(idata_h),
        },
        "uncertainty": unc,
    }
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("Sampler diagnostics:", json.dumps({k: v for k, v in diag.items() if k != "uncertainty"}))

    E.plot_forest_var(idata_b, "beta", out / "figures" / "forest_bayes_lr.png",
                      "Bayesian LR: coefficient posteriors", width=7.5)
    E.plot_forest_var(idata_h, "a", out / "figures" / "forest_group_intercepts.png",
                      "Hierarchical model: loan-purpose intercepts", width=7.0)
    E.plot_trace_vars(idata_h, ["mu_a", "sigma_a", "sigma_b"], out / "figures" / "trace_hierarchical.png")

    # SHAP for XGBoost vs. Bayesian coefficient magnitudes
    explainer = shap.TreeExplainer(xgbm)
    sv = explainer.shap_values(Xt_te)
    shap.summary_plot(sv, Xt_te, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / "figures" / "shap_xgboost.png", dpi=150)
    plt.close("all")

    shap_imp = pd.Series(np.abs(sv).mean(0), index=Xt_te.columns).sort_values(ascending=False)
    shap_imp.head(15).round(4).to_csv(out / "shap_top_features.csv", header=["mean_abs_shap"])
    beta_mean = E._ds(idata_b)["beta"].mean(("chain", "draw")).to_series().abs().sort_values(ascending=False)
    beta_mean.head(15).round(4).to_csv(out / "bayes_top_coefficients.csv", header=["abs_posterior_mean"])

    print(f"Done. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
