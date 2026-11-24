"""Churn labelling and snapshot (landmark) feature engineering.

Time convention: a snapshot taken on date ``s`` sees activity on days strictly
before ``s``. Durations count days from the start of day ``s``, so a churn that
becomes known at the end of day ``d`` has duration ``d - s + 1 >= 1``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from churn import config

PLAYER_COLUMNS = ("player_id", "registration_date", "age", *config.CATEGORICAL_FEATURES)
ACTIVITY_COLUMNS = (
    "player_id",
    "date",
    "sessions",
    "bets",
    "turnover",
    "ggr",
    "deposits",
    "withdrawals",
    "bonus",
)
_WINDOW_SUMS = ("sessions", "bets", "turnover", "ggr", "deposits", "withdrawals", "bonus")


def load_raw(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the players and activity tables.

    Parameters
    ----------
    data_dir : Path
        Directory holding ``players.parquet`` and ``activity.parquet``.

    Returns
    -------
    players, activity : pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    ValueError
        If a required column is missing or a player has two rows for one day.
    """
    players = pd.read_parquet(data_dir / "players.parquet")
    activity = pd.read_parquet(data_dir / "activity.parquet")
    for name, table, required in (
        ("players", players, PLAYER_COLUMNS),
        ("activity", activity, ACTIVITY_COLUMNS),
    ):
        missing = set(required) - set(table.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    if activity.duplicated(["player_id", "date"]).any():
        raise ValueError("activity table must have at most one row per player and day")
    return players, activity


def first_churn_dates(
    activity: pd.DataFrame,
    data_end: pd.Timestamp,
    inactivity_days: int = config.INACTIVITY_DAYS,
) -> pd.Series:
    """Date on which each player's first churn becomes known.

    Churn is the first run of ``inactivity_days`` consecutive inactive days and
    is dated at the end of that run (last activity + ``inactivity_days``), the
    first moment it can be observed without looking into the future.

    Parameters
    ----------
    activity : pd.DataFrame
        One row per player and active day, with ``player_id`` and ``date``.
    data_end : pd.Timestamp
        Last observed day (inclusive). Runs still open on that day do not count.
    inactivity_days : int
        Length of the inactivity run that defines churn.

    Returns
    -------
    pd.Series
        Churn date indexed by ``player_id``; players without churn are absent.

    Raises
    ------
    ValueError
        If ``activity`` contains days after ``data_end``.
    """
    if activity["date"].max() > data_end:
        raise ValueError("activity contains days after data_end")
    days = activity[["player_id", "date"]].sort_values(["player_id", "date"])
    # For a player's last active day the "next" day is the first unobserved one,
    # so a trailing run only counts once it is complete inside the data.
    next_day = days.groupby("player_id")["date"].shift(-1).fillna(data_end + pd.Timedelta(days=1))
    gap = (next_day - days["date"]).dt.days > inactivity_days
    last_before_gap = days.loc[gap].groupby("player_id")["date"].first()
    return (last_before_gap + pd.Timedelta(days=inactivity_days)).rename("churn_date")


def snapshot_features(
    players: pd.DataFrame,
    activity: pd.DataFrame,
    churn_dates: pd.Series,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Features of every player still at risk of churn on ``snapshot``.

    A player is at risk if they have played before ``snapshot`` and their first
    churn has not happened by the end of the previous day.

    Parameters
    ----------
    players, activity : pd.DataFrame
        Raw tables as returned by :func:`load_raw`.
    churn_dates : pd.Series
        Output of :func:`first_churn_dates`.
    snapshot : pd.Timestamp
        Snapshot date; only activity strictly before it is used.

    Returns
    -------
    pd.DataFrame
        ``player_id``, ``snapshot_date``, the columns in ``config.FEATURES`` and
        ``churn_date`` (label information, not a feature).
    """
    history = activity[activity["date"] < snapshot]
    last_active = history.groupby("player_id")["date"].max()
    churn = churn_dates.reindex(last_active.index)
    at_risk = last_active.index[(churn.isna() | (churn >= snapshot)).to_numpy()]

    # At-risk players have at least one active day in this window by definition.
    window = history[history["date"] >= snapshot - pd.Timedelta(days=config.INACTIVITY_DAYS)]
    sums = window.groupby("player_id")[list(_WINDOW_SUMS)].sum()
    active_30d = window.groupby("player_id").size()
    last_week = window[window["date"] >= snapshot - pd.Timedelta(days=7)]
    active_7d = last_week.groupby("player_id").size()

    static = players.set_index("player_id").reindex(at_risk)
    frame = pd.DataFrame(index=at_risk)
    frame["snapshot_date"] = snapshot
    frame["tenure_days"] = (snapshot - static["registration_date"]).dt.days
    frame["recency_days"] = (snapshot - last_active.reindex(at_risk)).dt.days
    frame["active_days_7d"] = active_7d.reindex(at_risk, fill_value=0)
    frame["active_days_30d"] = active_30d.reindex(at_risk)
    frame["activity_ratio_7d_30d"] = (frame["active_days_7d"] / 7) / (
        frame["active_days_30d"] / config.INACTIVITY_DAYS
    )
    for name in ("sessions", "turnover", "ggr", "deposits", "withdrawals", "bonus"):
        frame[f"{name}_30d"] = sums[name].reindex(at_risk)
    frame["avg_stake_30d"] = frame["turnover_30d"] / sums["bets"].reindex(at_risk)
    frame["age"] = static["age"]
    for name, categories in config.CATEGORICAL_FEATURES.items():
        frame[name] = pd.Categorical(static[name], categories=list(categories))
    frame["churn_date"] = churn_dates.reindex(at_risk)
    frame = frame.rename_axis("player_id").reset_index()
    return frame[["player_id", "snapshot_date", *config.FEATURES, "churn_date"]]


def build_snapshots(
    players: pd.DataFrame,
    activity: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    data_end: pd.Timestamp = config.DATA_END,
) -> pd.DataFrame:
    """Stack :func:`snapshot_features` over several snapshot dates.

    Parameters
    ----------
    players, activity : pd.DataFrame
        Raw tables as returned by :func:`load_raw`.
    snapshots : pd.DatetimeIndex
        Snapshot dates.
    data_end : pd.Timestamp
        Last observed day of ``activity`` (inclusive).

    Returns
    -------
    pd.DataFrame
        One row per at-risk player and snapshot.
    """
    churn_dates = first_churn_dates(activity, data_end)
    frames = [snapshot_features(players, activity, churn_dates, s) for s in snapshots]
    return pd.concat(frames, ignore_index=True)


def add_labels(
    frame: pd.DataFrame,
    censor_date: pd.Timestamp,
    max_follow_up: int = config.MAX_FOLLOW_UP_DAYS,
) -> pd.DataFrame:
    """Attach right-censored durations as they are known on ``censor_date``.

    Follow-up ends the day before ``censor_date`` or after ``max_follow_up``
    days, whichever comes first. A churn landing exactly on the last follow-up
    day is treated as censored: every event then lies strictly inside the
    follow-up window, which the IPCW metrics need (the estimated probability of
    being uncensored is zero on the last day).

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`build_snapshots`.
    censor_date : pd.Timestamp
        Date on which the labels are read; snapshots on or after it are dropped
        because they have no follow-up yet.
    max_follow_up : int
        Administrative cap on follow-up, in days.

    Returns
    -------
    pd.DataFrame
        Copy of the eligible rows with ``time`` (days, >= 1) and ``event``.
    """
    labelled = frame[frame["snapshot_date"] < censor_date].copy()
    days_known = (censor_date - labelled["snapshot_date"]).dt.days
    follow_up = np.minimum(days_known, max_follow_up)
    days_to_churn = (labelled["churn_date"] - labelled["snapshot_date"]).dt.days + 1
    event = days_to_churn < follow_up
    labelled["event"] = event.to_numpy()
    labelled["time"] = np.where(event, days_to_churn, follow_up).astype(float)
    return labelled
