# Casino churn survival

Predicts which online-casino players will churn within the next 60 days, using two
XGBoost survival models (Cox and AFT) on synthetic data, with a bilingual Streamlit
dashboard over the results.

## Quick start

```bash
uv sync
uv run python -m churn.generate_data        # writes data/players.parquet, data/activity.parquet
uv run python -m churn.pipeline             # trains, evaluates, writes artifacts/ (about 2 minutes)
uv run streamlit run dashboard/app.py       # English by default, Russian in the sidebar
uv run pytest -q
```

`churn.pipeline` accepts `--n-trials` (Optuna trials per model, default 40),
`--n-bootstrap` (default 200), `--seed`, `--data-dir` and `--artifacts-dir`.
Both scripts are seeded; re-running them reproduces the same data and artifacts.

## Notebook

`churn_survival.ipynb` walks through the same steps as the two scripts, one section per
requirement, with every table and chart inline. It is saved with its outputs, and running
it writes the same `data/` and `artifacts/` as the scripts do. The project environment
has `ipykernel` and `nbformat`, so an editor such as Cursor or VS Code can run it with
`.venv` as the kernel. JupyterLab and `nbclient` are not dependencies; `--with` supplies
them for one run:

```bash
uv run --with jupyterlab jupyter lab churn_survival.ipynb
uv run --with ipykernel --with nbclient jupyter-execute churn_survival.ipynb --inplace
```

Two more notebooks read the saved `data/` and `artifacts/` and write nothing:

- `predict_debug.ipynb` follows one prediction from the raw tables to the scorecard and
  prints every intermediate array with its shape and meaning, checking each step against
  the library.
- `predict_players.ipynb` scores one or several players by `player_id` on a chosen date.

## Problem setup

- **Churn** is the first run of 30 consecutive days without activity, dated on the
  30th inactive day. A player leaves the population after the first churn.
- **Snapshots.** On the first day of each month every player still at risk becomes one
  row: features use activity strictly before the snapshot, the target is the number of
  days until churn, followed for at most 90 days and censored at the label cut-off.
- **Temporal split.** Data cover 2024-01-01 .. 2026-06-30 for 20,000 players.

  | Set | Snapshots | Labels known up to |
  | --- | --- | --- |
  | Train | 2024-07-01 .. 2025-12-01 | 2026-01-01 |
  | Validation | 2026-01-01 | 2026-04-01 |
  | Test | 2026-04-01 | 2026-07-01 |
  | Scoring | 2026-07-01 | no labels |

- **Models.** `survival:cox` with a Breslow baseline hazard and `survival:aft`; both
  expose `predict_survival(X, times)` on the day grid 1..85.

## What the pipeline does

1. Tunes both models with Optuna, minimising the integrated Brier score on validation,
   and picks the best model by that score.
2. Calibrates both models with per-day isotonic regression fitted on validation.
3. Evaluates on test: C-index, IBS, Brier score and AUC over time, predicted vs observed
   survival, decile calibration and daily hazard, with a paired bootstrap over players.
4. Explains all models with gain importance and SHAP values (dependence plots).
5. Compares the remaining useful life of the best model with a Kaplan-Meier baseline.
6. Backtests both models on a rolling origin (11 monthly folds).
7. Refits the best model on train + validation as the production model and compares it
   with the evaluation model on test.
8. Scores the players active on the scoring date: churn probability at 7/14/30/60 days,
   remaining useful life, risk segment and the top three SHAP drivers per player.

Everything the dashboard needs is saved to `artifacts/`: `metrics.json`, one parquet
table per chart, and the boosters and calibrators under `artifacts/models/`.

## Layout

```
churn/config.py          dates, horizons, feature lists
churn/generate_data.py   synthetic players and daily activity
churn/features.py        churn dates, snapshot features, survival labels
churn/models.py          XGBoost Cox and AFT with a common survival interface
churn/evaluation.py      metrics, calibration, bootstrap, hazard and RUL tables
churn/pipeline.py        end-to-end run and artifact export
churn_survival.ipynb     the same run as a notebook with inline charts
predict_debug.ipynb      one prediction step by step: calls, arrays, shapes, checks
predict_players.ipynb    prediction for selected players on a chosen date
dashboard/app.py         Streamlit dashboard; dashboard/i18n.py holds the translations
tests/                   pytest suite for features, models and evaluation
```

## Caveats

- The data are synthetic, so the reported quality says nothing about real players.
- A player inactive for `r` days churns for certain in `31 - r` days unless they come
  back, so recency dominates the models and the riskiest players are simply the longest
  inactive ones. The dashboard has a recency filter to look past them.
- Remaining useful life is restricted to the 85-day grid: it is `E[min(T, 85)]`.
- Backtest folds reuse the hyperparameters tuned on the main split and are not
  calibrated.
- The production model is trained on snapshots up to 2026-01-01 and scores 2026-07-01;
  its calibrator is the one fitted for the evaluation model.
- Cox training uses `max_delta_step` to keep the margins finite: near-deterministic
  churn of long-inactive players otherwise drives the partial likelihood to overflow.
