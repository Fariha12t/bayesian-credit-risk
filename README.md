# Bayesian Credit Risk Modelling

Bayesian logistic regression and a hierarchical model for credit default, compared against an XGBoost baseline on the public **German Credit** dataset (UCI / OpenML `credit-g`, 1,000 applicants).

The question this project answers: **a single accuracy number (for example, 97.8% in my XGBoost paper) says nothing about how sure the model is about each applicant.** A Bayesian treatment gives a full posterior over default probability for every applicant, so we can see calibration, credible intervals, and where the model should abstain.

## What is in here

| Model | Purpose |
|---|---|
| XGBoost | Baseline, explained with SHAP |
| Logistic regression (MLE) | Frequentist reference |
| Bayesian logistic regression | Weakly-informative Normal priors, full posterior over all coefficients |
| Hierarchical Bayesian logistic regression | Partial pooling across **loan purpose** (varying intercept and varying slope on loan duration), non-centred parametrisation |

Evaluation covers accuracy, AUC, Brier score, log loss, expected calibration error (ECE), calibration curves, per-applicant credible intervals, and a selective-prediction analysis ("if the model abstains on its most uncertain applicants, does accuracy on the rest improve?").

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_all --data german --out results
```

Runtime is a few minutes on a laptop. Use `--draws 500 --tune 500 --chains 2` for a quick run.

`--data synthetic` generates a fake dataset with the same schema. It is only for checking that the pipeline runs offline; do not report its numbers.

## Outputs (`results/`)

- `metrics.csv`: accuracy, AUC, Brier, log loss, ECE for every model
- `diagnostics.json`: R-hat, effective sample size, divergences, uncertainty diagnostics
- `posterior_summary_*.csv`: posterior means, standard deviations and 94% intervals
- `selective_prediction.csv`: accuracy versus coverage when abstaining on uncertain applicants
- `shap_top_features.csv` and `bayes_top_coefficients.csv`: feature importance, SHAP versus posterior coefficients
- `figures/`: `calibration.png`, `credible_intervals.png`, `selective_prediction.png`, `forest_bayes_lr.png`, `forest_group_intercepts.png`, `trace_hierarchical.png`, `shap_xgboost.png`

## Results

> Fill this section in after running on the German Credit data. Suggested structure:
>
> 1. Metrics table (paste `results/metrics.csv`).
> 2. Calibration: which model is better calibrated, and by how much (ECE, curve).
> 3. Uncertainty: error rate in the high-uncertainty half versus the low-uncertainty half of applicants, and the selective-prediction plot.
> 4. Hierarchical model: do loan purposes differ (`sigma_a`)? Which purposes shift risk most?
> 5. Agreement between SHAP rankings and posterior coefficients.
> 6. Sampler health: max R-hat below 1.01, no divergences.

## Project layout

```
src/data.py       data loading and preprocessing (German Credit + synthetic fallback)
src/models.py     PyMC models, posterior prediction, XGBoost
src/evaluate.py   metrics, calibration, selective prediction, plots
src/run_all.py    end-to-end pipeline
results/          generated outputs (git-ignored by default)
```

## Method notes

- Numeric features are standardised; `credit_amount` is log-transformed. Categorical features are one-hot encoded (first level dropped for the linear models).
- Priors: intercept `Normal(0, 2.5)`, coefficients `Normal(0, 1)` on standardised inputs. Hierarchical scales use `HalfNormal`.
- Posterior predictive probabilities are computed per applicant over all draws; "uncertainty" is the posterior standard deviation of the default probability.
- 70/30 stratified train/test split, fixed seed.

## Limitations

- German Credit is small (1,000 rows). The Bayesian and frequentist estimates will be close; the value is in the uncertainty, not in beating XGBoost on accuracy.
- Posterior uncertainty here is parameter uncertainty under the model. It does not capture model misspecification.
- The same pipeline can be pointed at Home Credit or Give Me Some Credit by writing a loader that returns a `CreditData` object (see `src/data.py`). Those datasets are much larger, so use a subsample or variational inference for the hierarchical model.



