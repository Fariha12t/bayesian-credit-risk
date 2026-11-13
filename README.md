# Data

The pipeline downloads the German Credit dataset automatically from OpenML
(`credit-g`, version 1) the first time you run it. Nothing needs to be done here
in the normal case.

## Offline fallback

If OpenML is unreachable, `src/data.py` reads `data/credit.csv` instead.
This file is not redistributed in the repo (it is listed in `.gitignore`).

To create it, run once on a machine with internet access:

```bash
python -c "from sklearn.datasets import fetch_openml; fetch_openml('credit-g', version=1, as_frame=True).frame.to_csv('data/credit.csv', index=False)"
```

The CSV must keep the OpenML column names (`checking_status`, `duration`,
`credit_history`, `purpose`, `credit_amount`, ...) and the target column `class`
with values `good` / `bad`.

Source: UCI Statlog (German Credit Data), https://www.openml.org/d/31
