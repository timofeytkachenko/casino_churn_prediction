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

# --- Temporal split (snapshot dates; features use history strictly before) ---
TRAIN_SNAPSHOTS = pd.date_range("2024-07-01", "2025-12-01", freq="MS")
VAL_SNAPSHOT = pd.Timestamp("2026-01-01")
TEST_SNAPSHOT = pd.Timestamp("2026-04-01")
SCORING_SNAPSHOT = DATA_END + pd.Timedelta(days=1)
BACKTEST_SNAPSHOTS = pd.date_range("2025-07-01", "2026-05-01", freq="MS")
ALL_SNAPSHOTS = pd.date_range(TRAIN_SNAPSHOTS[0], SCORING_SNAPSHOT, freq="MS")

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

# --- Scoring ----------------------------------------------------------------
# Lower bounds on P(churn within HORIZON_DAYS), checked from the top down.
RISK_SEGMENTS = (("high", 0.70), ("medium", 0.40), ("low", 0.0))
TOP_N_RISKY = 100
