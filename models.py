"""Bayesian logistic regression, hierarchical logistic regression and XGBoost baseline."""
from __future__ import annotations

import numpy as np
import pymc as pm
import xgboost as xgb


def _posterior(idata):
    """Return the posterior as an xarray Dataset (works for ArviZ 0.x InferenceData and 1.x DataTree)."""
    post = idata.posterior
    return getattr(post, "dataset", post)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _sample(model, draws, tune, chains, seed):
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
        )


def fit_bayes_lr(X: np.ndarray, y: np.ndarray, feature_names, draws=1000, tune=1000, chains=4, seed=42):
    """Bayesian logistic regression with weakly-informative Normal priors (features are standardised)."""
    coords = {"feature": list(feature_names)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0, 2.5)
        beta = pm.Normal("beta", 0, 1.0, dims="feature")
        logit = alpha + pm.math.dot(X, beta)
        pm.Bernoulli("y", logit_p=logit, observed=y)
    return _sample(model, draws, tune, chains, seed)


def predict_bayes_lr(idata, X: np.ndarray) -> np.ndarray:
    """Posterior predictive default probabilities, shape (n_draws, n_obs)."""
    post = _posterior(idata).stack(s=("chain", "draw"))
    alpha = post["alpha"].values  # (S,)
    beta = post["beta"].values  # (F, S)
    return _sigmoid(alpha[:, None] + (X @ beta).T)


def fit_hier_lr(
    X: np.ndarray,
    group: np.ndarray,
    y: np.ndarray,
    group_names,
    feature_names,
    draws=1000,
    tune=1000,
    chains=4,
    seed=42,
):
    """Hierarchical logistic regression (non-centred).

    Partial pooling over loan purpose: a varying intercept and a varying slope on the
    first feature (loan duration). All other coefficients are shared.
    """
    G = len(group_names)
    coords = {"feature": list(feature_names), "group": list(group_names)}
    with pm.Model(coords=coords) as model:
        mu_a = pm.Normal("mu_a", 0, 2.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a_off = pm.Normal("a_off", 0, 1, dims="group")
        a = pm.Deterministic("a", mu_a + sigma_a * a_off, dims="group")

        sigma_b = pm.HalfNormal("sigma_b", 0.5)
        b_off = pm.Normal("b_off", 0, 1, dims="group")
        beta = pm.Normal("beta", 0, 1.0, dims="feature")
        # varying slope = shared slope on feature 0 + group deviation
        b_dur = pm.Deterministic("b_dur", beta[0] + sigma_b * b_off, dims="group")

        logit = a[group] + pm.math.dot(X[:, 1:], beta[1:]) + b_dur[group] * X[:, 0]
        pm.Bernoulli("y", logit_p=logit, observed=y)
    return _sample(model, draws, tune, chains, seed)


def predict_hier_lr(idata, X: np.ndarray, group: np.ndarray) -> np.ndarray:
    post = _posterior(idata).stack(s=("chain", "draw"))
    a = post["a"].values  # (G, S)
    b_dur = post["b_dur"].values  # (G, S)
    beta = post["beta"].values  # (F, S)
    lin = (X[:, 1:] @ beta[1:]).T  # (S, n)
    logit = a[group].T + lin + b_dur[group].T * X[:, 0][None, :]
    return _sigmoid(logit)


def fit_xgb(X, y, seed=42):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X, y)
    return model
