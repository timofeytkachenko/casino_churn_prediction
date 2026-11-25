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

- **Decisions before the test.** Every decision is taken on rolling-origin folds: the
  models are fitted on the snapshots before the fold date with the labels known on that
  date, and forecast the players at risk on it. All their labels are read on
  2026-04-01 at the latest.

  | Decision | Fold dates |
  | --- | --- |
  | Hyperparameters | 2025-07-01, 2025-10-01 |
  | Model family, calibrated or raw | 2025-11-01, 2025-12-01 |

- **Models.** `survival:cox` with a Breslow baseline hazard and `survival:aft`; both
  expose `predict_survival(X, times)` on the day grid 1..85.
- **Time origin.** A player last active `r` days before a snapshot cannot churn during
  the first `30 - r` days. The models are trained on the time left after that delay
  (`churn_free_days`, which is not a feature) and return `S = 1` before it, so no
  probability is placed on days when churn is impossible.

## What the pipeline does

1. Tunes both models with Optuna, minimising the integrated Brier score averaged over
   the tuning folds.
2. Chooses the model on the selection folds, forecast the way production works: models
   refitted on all earlier snapshots, calibrator taken from the model fitted three
   months earlier. A family is calibrated only if the 95% interval of the calibration
   effect lies below zero; the family with the lower integrated Brier score wins.
   Intervals come from a bootstrap over players.
3. Evaluates on test the models trained on the train snapshots, raw and calibrated on
   validation: C-index, IBS, Brier score and AUC over time, predicted vs observed
   survival, decile calibration and daily hazard, with a paired bootstrap over players.
   Two references use no model: Kaplan-Meier, and Kaplan-Meier by recency (what the
   churn definition alone gives). The metrics are also reported within recency strata.
4. Explains all models with gain importance and SHAP values (dependence plots).
5. Compares the remaining useful life of the best model with a Kaplan-Meier baseline.
6. Backtests both models, raw and calibrated, on a rolling origin (11 monthly folds).
7. Refits the selected family on every snapshot before the scoring date as the
   production model, with the calibrator of the model fitted three months earlier, and
   replays that procedure on the test date next to the evaluation model.
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
tests/                   pytest suite for features, models, evaluation and the pipeline
```

## Caveats

- The data are synthetic, so the reported quality says nothing about real players.
- A player inactive for `r` days churns for certain in `31 - r` days unless they come
  back, so recency dominates the forecast and the riskiest players are simply the
  longest inactive ones. The dashboard has a recency filter to look past them, and the
  quality section reports the metrics within recency strata.
- The risk score is the booster margin on the model clock: it ranks players by how soon
  they churn once they can, not by churn at a fixed day. Rank by the churn probability.
- Remaining useful life is restricted to the 85-day grid: it is `E[min(T, 85)]`.
- Hyperparameters are tuned once. Backtest folds on or before the last tuning fold
  (2025-10-01) therefore see hyperparameters chosen with their own outcomes, and the
  outcome window of that tuning fold overlaps in calendar time with the outcome windows
  of the selection folds. Only the test date and the dates after it are untouched by
  every decision.
- The selection folds are two snapshots of largely the same players, one month apart.
  The bootstrap resamples players, but the interval still describes this population
  and these two months, not a different period.
- The production model cannot be tested: the outcomes of the players it scores are in
  the future. Its test-date replay is the evidence for the procedure. Its calibrator is
  fitted on the test snapshot, after the test report, and is three months older than
  the model it calibrates.
- Cox training uses `max_delta_step` to keep the margins finite: near-deterministic
  churn of long-inactive players otherwise drives the partial likelihood to overflow.
