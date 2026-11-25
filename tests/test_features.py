"""Tests for churn labelling and snapshot features."""

import numpy as np
import pandas as pd
import pytest

from churn import config, features
from churn.generate_data import generate


def _activity(days_by_player: dict[int, list[str]]) -> pd.DataFrame:
    """Activity table with unit volumes on the given player-days."""
    rows = [(pid, pd.Timestamp(day)) for pid, days in days_by_player.items() for day in days]
    activity = pd.DataFrame(rows, columns=["player_id", "date"])
    activity[["sessions", "bets"]] = 1
    activity[["turnover", "ggr", "deposits", "withdrawals", "bonus"]] = 1.0
    return activity


def _players(ids: list[int], registered: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ids,
            "registration_date": pd.Timestamp(registered),
            "age": 30,
            "acquisition_channel": "organic",
            "favorite_game": "slots",
        }
    )


@pytest.fixture(scope="module")
def simulated() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    end = pd.Timestamp("2024-12-31")
    players, activity = generate(n_players=400, start=pd.Timestamp("2024-01-01"), end=end, seed=1)
    return players, activity, end


def test_first_churn_dates_uses_first_completed_inactivity_run():
    activity = _activity(
        {
            1: ["2025-01-01", "2025-01-03", "2025-02-10"],  # 37 idle days after Jan 3
            2: ["2025-01-01", "2025-01-31"],  # 29 idle days: not a churn
            3: ["2025-01-16"],  # run completes exactly on data_end
            4: ["2025-01-17"],  # run still open on data_end
        }
    )
    churn = features.first_churn_dates(activity, data_end=pd.Timestamp("2025-02-15"))
    assert churn.to_dict() == {1: pd.Timestamp("2025-02-02"), 3: pd.Timestamp("2025-02-15")}


def test_first_churn_dates_rejects_activity_after_data_end():
    activity = _activity({1: ["2025-01-10"]})
    with pytest.raises(ValueError, match="after data_end"):
        features.first_churn_dates(activity, data_end=pd.Timestamp("2025-01-09"))


def test_snapshot_features_windows_and_eligibility():
    activity = _activity(
        {
            1: ["2025-02-01", "2025-02-20", "2025-02-27", "2025-03-01"],  # Mar 1 is not history
            2: ["2025-01-01"],  # churned on Jan 31, before the snapshot
            3: ["2025-03-05"],  # first plays after the snapshot
        }
    )
    players = _players([1, 2, 3], registered="2025-01-01")
    churn = features.first_churn_dates(activity, data_end=pd.Timestamp("2025-03-31"))
    frame = features.snapshot_features(players, activity, churn, pd.Timestamp("2025-03-01"))

    assert frame["player_id"].tolist() == [1]
    row = frame.iloc[0]
    assert row["tenure_days"] == 59
    assert row["recency_days"] == 2
    assert row[config.DELAY_COLUMN] == 28  # the 30-day run can end on day 29 at the earliest
    assert config.DELAY_COLUMN not in config.FEATURES
    assert row["active_days_7d"] == 1
    assert row["active_days_30d"] == 3
    assert row["activity_ratio_7d_30d"] == pytest.approx((1 / 7) / (3 / 30))
    assert row["turnover_30d"] == 3.0
    assert row["avg_stake_30d"] == 1.0


def test_add_labels_censoring_rules():
    snapshot = pd.Timestamp("2025-01-01")
    frame = pd.DataFrame(
        {
            "snapshot_date": [snapshot] * 5 + [pd.Timestamp("2025-06-01")],
            "churn_date": pd.to_datetime(
                ["2025-01-01", "2025-01-10", "2025-03-30", "2025-03-31", None, None]
            ),
        }
    )

    full = features.add_labels(frame, censor_date=pd.Timestamp("2025-06-01"))
    assert len(full) == 5  # the snapshot taken on the censor date has no follow-up
    assert full["time"].tolist() == [1, 10, 89, 90, 90]
    assert full["event"].tolist() == [True, True, True, False, False]  # day 90 is censored

    early = features.add_labels(frame, censor_date=pd.Timestamp("2025-01-10"))
    assert early["time"].tolist() == [1, 9, 9, 9, 9]  # Jan 10 churn is not known yet
    assert early["event"].tolist() == [True, False, False, False, False]

    # Known on the last follow-up day, but still censored (tie with the censoring time).
    tie = features.add_labels(frame, censor_date=pd.Timestamp("2025-01-11"))
    assert tie["time"].tolist() == [1, 10, 10, 10, 10]
    assert tie["event"].tolist() == [True, False, False, False, False]


def test_features_and_labels_do_not_look_ahead(simulated):
    """Truncating the raw data at the cutoff must not change anything before it."""
    players, activity, end = simulated
    cutoff = pd.Timestamp("2024-09-01")
    snapshots = pd.date_range("2024-04-01", "2024-08-01", freq="MS")

    with_future = features.build_snapshots(players, activity, snapshots, data_end=end)
    without_future = features.build_snapshots(
        players,
        activity[activity["date"] < cutoff],
        snapshots,
        data_end=cutoff - pd.Timedelta(days=1),
    )
    columns = ["player_id", "snapshot_date", *config.FEATURES, "time", "event"]
    pd.testing.assert_frame_equal(
        features.add_labels(with_future, cutoff)[columns],
        features.add_labels(without_future, cutoff)[columns],
    )


def test_at_risk_invariants(simulated):
    players, activity, end = simulated
    snapshots = pd.date_range("2024-04-01", "2024-12-01", freq="MS")
    frame = features.build_snapshots(players, activity, snapshots, data_end=end)
    labelled = features.add_labels(frame, end + pd.Timedelta(days=1))

    assert frame["recency_days"].between(1, config.INACTIVITY_DAYS).all()
    assert not frame.duplicated(["player_id", "snapshot_date"]).any()
    # Nobody is at risk after their first churn.
    assert (frame["churn_date"].isna() | (frame["churn_date"] >= frame["snapshot_date"])).all()
    assert (labelled["time"] >= 1).all()
    assert labelled.loc[labelled["event"], "time"].max() < config.MAX_FOLLOW_UP_DAYS
    # A player who never returns churns INACTIVITY_DAYS after the last active day.
    events = labelled[labelled["event"]]
    assert np.all(events["time"] + events["recency_days"] >= config.INACTIVITY_DAYS + 1)
    # The same bound as the delay the models start their clock from.
    assert (frame[config.DELAY_COLUMN] == config.INACTIVITY_DAYS - frame["recency_days"]).all()
    assert (events["time"] > events[config.DELAY_COLUMN]).all()
    assert (events["time"] == events[config.DELAY_COLUMN] + 1).any()
