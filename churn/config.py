"""Central configuration: calendar, horizons, feature schema and paths."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42

# --- Synthetic data ---------------------------------------------------------
N_PLAYERS = 20_000
DATA_START = pd.Timestamp("2024-01-01")
DATA_END = pd.Timestamp("2026-06-30")  # last observed day, inclusive
CHANNELS = ("organic", "affiliate", "paid_ads", "referral")
GAMES = ("slots", "live_casino", "table_games", "crash_games")

# --- Churn definition and time axis -----------------------------------------
# A player churns on the day that completes the first run of INACTIVITY_DAYS
# consecutive days without activity, i.e. last activity + INACTIVITY_DAYS.
INACTIVITY_DAYS = 30
HORIZON_DAYS = 60  # business horizon: P(churn within the next 60 days)
MAX_FOLLOW_UP_DAYS = 90  # administrative censoring of every label
# Survival curves are evaluated on a daily grid that stays strictly inside the
# follow-up window, which is what the IPCW-based metrics require.
TIME_GRID = np.arange(1, 86)
SCORE_HORIZONS = (7, 14, 30, 60)
# A player last active r days before a snapshot cannot churn during the first
# INACTIVITY_DAYS - r days after it. The models count time from the end of that
# delay; the column carries the delay and is never used as a feature.
DELAY_COLUMN = "churn_free_days"

# --- Temporal split (snapshot dates; features use history strictly before) ---
TRAIN_SNAPSHOTS = pd.date_range("2024-07-01", "2025-12-01", freq="MS")
VAL_SNAPSHOT = pd.Timestamp("2026-01-01")
TEST_SNAPSHOT = pd.Timestamp("2026-04-01")
SCORING_SNAPSHOT = DATA_END + pd.Timedelta(days=1)
BACKTEST_SNAPSHOTS = pd.date_range("2025-07-01", "2026-05-01", freq="MS")
ALL_SNAPSHOTS = pd.date_range(TRAIN_SNAPSHOTS[0], SCORING_SNAPSHOT, freq="MS")
# Rolling-origin folds behind the decisions taken before the test snapshot. Each
# fold is forecast from the snapshots before it and is followed for at most
# MAX_FOLLOW_UP_DAYS, so none of their labels reaches the test snapshot.
TUNE_SNAPSHOTS = pd.DatetimeIndex(["2025-07-01", "2025-10-01"])  # hyperparameters
SELECTION_SNAPSHOTS = pd.DatetimeIndex(["2025-11-01", "2025-12-01"])  # family, calibration
# A calibrator is fitted on a snapshot once it has been followed for the whole
# time grid, so a forecast uses the calibrator of the model fitted this much earlier.
CALIBRATION_LAG = pd.DateOffset(months=3)

# --- Feature schema ---------------------------------------------------------
NUMERIC_FEATURES = (
    "tenure_days",
    "recency_days",
    "active_days_7d",
    "active_days_30d",
    "activity_ratio_7d_30d",
    "sessions_30d",
    "turnover_30d",
    "ggr_30d",
    "deposits_30d",
    "withdrawals_30d",
    "bonus_30d",
    "avg_stake_30d",
    "age",
)
CATEGORICAL_FEATURES = {"acquisition_channel": CHANNELS, "favorite_game": GAMES}
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# --- Evaluation -------------------------------------------------------------
# Inclusive bounds on recency_days. Long-inactive players churn almost surely
# and dominate pooled metrics, so quality is also reported within these strata.
RECENCY_STRATA = ((1, 3), (4, 7), (8, INACTIVITY_DAYS))

# --- Scoring ----------------------------------------------------------------
# Lower bounds on P(churn within HORIZON_DAYS), checked from the top down.
RISK_SEGMENTS = (("high", 0.70), ("medium", 0.40), ("low", 0.0))
TOP_N_RISKY = 100
