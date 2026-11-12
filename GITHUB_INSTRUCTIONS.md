# Adding this project to GitHub

## Option A: command line

1. Unzip `bayesian-credit-risk.zip` and open a terminal in the folder.
2. On GitHub, click **New repository**. Name it `bayesian-credit-risk`. Leave it empty (no README, no .gitignore, no license).
3. Run:

```bash
cd bayesian-credit-risk
git init
git add .
git commit -m "Add Bayesian credit risk modelling project"
git branch -M main
git remote add origin https://github.com/Fariha12t/bayesian-credit-risk.git
git push -u origin main
```

## Option B: browser only

1. Create the empty repository as above.
2. Click **uploading an existing file**.
3. Drag in the *contents* of the unzipped folder (including the `src` folder and `.gitignore`).
4. Click **Commit changes**.

## Before you publish: run it and add results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all --data german --out results
```

Then:

1. Fill in the **Results** section of `README.md` with the numbers from `results/metrics.csv` and `results/diagnostics.json`.
2. Copy the key figures somewhere that gets committed, for example `docs/`:
   ```bash
   mkdir docs && cp results/figures/calibration.png results/figures/selective_prediction.png results/figures/credible_intervals.png docs/
   ```
   Then embed them in the README with `![Calibration](docs/calibration.png)`.
3. Commit and push:
   ```bash
   git add README.md docs
   git commit -m "Add results and figures"
   git push
   ```

## Make the repo stand out

- Add a one-line description and topics on the repo page: `bayesian-inference`, `pymc`, `credit-risk`, `xgboost`, `shap`, `uncertainty-quantification`.
- Pin the repo on your GitHub profile.
- Link your Springer paper in the README so the extension is obvious.
- If you like, add a `notebooks/` folder with a walkthrough notebook that calls the functions in `src/`.
