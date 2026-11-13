"""Metrics, calibration and uncertainty analysis."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return ece


def metrics_row(name, y, p, threshold=0.5):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "accuracy": accuracy_score(y, p >= threshold),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def summarise_draws(draws: np.ndarray, hdi=0.94):
    """draws: (S, n). Returns mean prob, std, and equal-tailed credible interval width."""
    lo, hi = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    return {
        "mean": draws.mean(0),
        "std": draws.std(0),
        "lo": np.quantile(draws, lo, axis=0),
        "hi": np.quantile(draws, hi, axis=0),
    }


def plot_calibration(y, probs: dict, path, bins=8):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for name, p in probs.items():
        frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Calibration curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def selective_prediction(y, p_mean, uncertainty, p_xgb, coverages=None):
    """Accuracy / log-loss when the model abstains on its most uncertain applicants.

    Bayesian: abstain on the largest posterior std. XGBoost: abstain on the least
    confident predictions (closest to 0.5), the only 'uncertainty' a point estimate has.
    """
    coverages = coverages if coverages is not None else np.linspace(0.3, 1.0, 15)
    rows = []
    order_b = np.argsort(uncertainty)  # most certain first
    order_x = np.argsort(-np.abs(p_xgb - 0.5))
    for c in coverages:
        k = max(int(round(c * len(y))), 5)
        ib, ix = order_b[:k], order_x[:k]
        rows.append(
            {
                "coverage": c,
                "bayes_accuracy": accuracy_score(y[ib], p_mean[ib] >= 0.5),
                "xgb_accuracy": accuracy_score(y[ix], p_xgb[ix] >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def plot_selective(df, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["coverage"], df["bayes_accuracy"], marker="o", label="Bayesian (abstain on high posterior std)")
    ax.plot(df["coverage"], df["xgb_accuracy"], marker="s", label="XGBoost (abstain on |p-0.5| small)")
    ax.set_xlabel("Coverage (fraction of applicants decided automatically)")
    ax.set_ylabel("Accuracy on retained applicants")
    ax.set_title("Does uncertainty tell you where to be careful?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_interval_examples(y, summ, path, n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=min(n, len(y)), replace=False))
    order = idx[np.argsort(summ["mean"][idx])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(order))
    ax.errorbar(
        x,
        summ["mean"][order],
        yerr=[summ["mean"][order] - summ["lo"][order], summ["hi"][order] - summ["mean"][order]],
        fmt="o",
        ms=3,
        capsize=2,
        label="Posterior mean and 94% interval",
    )
    ax.scatter(x, y[order], marker="x", c="crimson", s=20, label="Actual outcome (0/1)")
    ax.set_xlabel("Test applicants (sorted by predicted risk)")
    ax.set_ylabel("Default probability")
    ax.set_title("Per-applicant credible intervals (Bayesian LR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def interval_coverage(y, p_mean, std, p_xgb):
    """Does a high posterior std flag applicants that are misclassified more often?"""
    wrong_b = (p_mean >= 0.5) != y
    q = np.quantile(std, [0.5])[0]
    hi, lo = std >= q, std < q
    return {
        "error_rate_high_uncertainty_half": float(wrong_b[hi].mean()),
        "error_rate_low_uncertainty_half": float(wrong_b[lo].mean()),
        "spearman_std_vs_error": float(pd.Series(std).corr(pd.Series(wrong_b.astype(float)), method="spearman")),
    }


# ---------------------------------------------------------------------------
# Version-proof posterior helpers (ArviZ 0.x InferenceData and 1.x DataTree)
# ---------------------------------------------------------------------------
def _ds(idata, group="posterior"):
    g = getattr(idata, group)
    return getattr(g, "dataset", g)


def arviz_summary(idata, var_names, prob=0.94):
    import arviz as az

    try:
        return az.summary(idata, var_names=var_names, ci_prob=prob)
    except TypeError:
        return az.summary(idata, var_names=var_names, hdi_prob=prob)


def n_divergences(idata) -> int:
    return int(_ds(idata, "sample_stats")["diverging"].sum())


def plot_forest_var(idata, var, path, title, prob=0.94, width=7.0):
    """Forest plot of a 1-D posterior variable (mean, 50% and `prob` intervals)."""
    da = _ds(idata)[var].stack(s=("chain", "draw"))
    dim = [d for d in da.dims if d != "s"][0]
    labels = [str(v) for v in da[dim].values]
    vals = da.transpose(dim, "s").values
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    q = np.quantile(vals, [lo, 0.25, 0.5, 0.75, hi], axis=1)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(width, 0.28 * len(labels) + 1.5))
    ax.hlines(y, q[0], q[4], color="tab:blue", lw=1)
    ax.hlines(y, q[1], q[3], color="tab:blue", lw=3)
    ax.plot(q[2], y, "o", color="white", mec="tab:blue", ms=4)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(f"Posterior (thick = 50%, thin = {int(prob*100)}% interval)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trace_vars(idata, var_names, path):
    ds = _ds(idata)
    fig, axes = plt.subplots(len(var_names), 2, figsize=(9, 2.2 * len(var_names)))
    for i, v in enumerate(var_names):
        arr = ds[v].values  # (chain, draw)
        for c in range(arr.shape[0]):
            axes[i, 0].plot(arr[c], lw=0.6, alpha=0.8)
            axes[i, 1].hist(arr[c], bins=30, alpha=0.5, density=True)
        axes[i, 0].set_ylabel(v)
    axes[0, 0].set_title("Trace (one line per chain)")
    axes[0, 1].set_title("Marginal posterior per chain")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
