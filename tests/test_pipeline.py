"""Tests for the rolling-origin protocol: tuning, generations, backtest, selection, scoring."""

import numpy as np
import pandas as pd
import pytest

from churn import config, features, pipeline
from churn import evaluation as ev
from churn.generate_data import generate

TIMES = config.TIME_GRID
HORIZON = config.HORIZON_DAYS
DATA_END = pd.Timestamp("2024-12-31")
CUTOFF = DATA_END + pd.Timedelta(days=1)
DATES = pd.DatetimeIndex(["2024-09-01", "2024-10-01"])  # followed for the whole grid by CUTOFF
TUNED = {
    "cox": {"num_boost_round": 30, "learning_rate": 0.1, "max_depth": 2},
    "aft": {"num_boost_round": 30, "learning_rate": 0.1, "max_depth": 2},
}


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    players, activity = generate(
        n_players=1500, start=pd.Timestamp("2024-01-01"), end=DATA_END, seed=2
    )
    dates = pd.date_range("2024-03-01", "2024-12-01", freq="MS")
    return features.build_snapshots(players, activity, dates, data_end=DATA_END)


@pytest.fixture(scope="module")
def generations(snapshots) -> pipeline.Generations:
    return pipeline.Generations(snapshots, TUNED, seed=0)


def _impossible(frame: pd.DataFrame) -> np.ndarray:
    """Mask (players, days) of the days on which a player cannot have churned yet."""
    return TIMES[None, :] <= frame[config.DELAY_COLUMN].to_numpy()[:, None]


def test_generation_sees_only_what_was_known_on_its_date(generations):
    date = DATES[0]
    history = generations.history(date)
    assert (history["snapshot_date"] < date).all()
    last_followed_day = history["snapshot_date"] + pd.to_timedelta(history["time"], unit="D")
    assert (last_followed_day <= date).all()
    assert history["snapshot_date"].max() == date - pd.DateOffset(months=1)

    model = generations.model("cox", date)
    assert generations.model("cox", date) is model  # fitted once
    assert model.delay_column == config.DELAY_COLUMN
    assert model.feature_names_ == list(config.FEATURES)


def test_forecast_is_calibrated_by_the_lagged_generation(generations, snapshots):
    date = DATES[1]
    fold = pipeline.forecast(generations, date, CUTOFF)
    assert set(fold.surv_by_model) == {"cox_raw", "cox_calibrated", "aft_raw", "aft_calibrated"}

    frame = snapshots[snapshots["snapshot_date"] == date]
    np.testing.assert_array_equal(fold.cluster, frame["player_id"])
    raw = generations.model("cox", date).predict_survival(frame[pipeline.MODEL_COLUMNS], TIMES)
    np.testing.assert_array_equal(fold.surv_by_model["cox_raw"], raw)

    # The calibrator: the generation three months back, on its own snapshot, with
    # the outcomes known on the forecast date and not a day later.
    lagged = date - config.CALIBRATION_LAG
    _, lagged_fold = pipeline.rolling_fold(snapshots, lagged, date)
    X, time, event = pipeline.xy(lagged_fold)
    lagged_surv = generations.model("cox", lagged).predict_survival(X, TIMES)
    calibrator = ev.IsotonicSurvivalCalibrator().fit(lagged_surv, time, event, TIMES)
    np.testing.assert_allclose(
        fold.surv_by_model["cox_calibrated"], calibrator.transform(raw, TIMES)
    )

    # No variant of any model lets a player churn before it is possible.
    impossible = _impossible(frame)
    assert impossible.any()
    for surv in fold.surv_by_model.values():
        assert np.all(surv[impossible] == 1.0)


def test_backtest_reports_every_date_model_and_variant(generations):
    table = pipeline.backtest(generations, CUTOFF, DATES)
    assert len(table) == len(DATES) * 2 * 2
    assert not table.duplicated(["snapshot_date", "model", "variant"]).any()
    assert set(table["variant"]) == set(pipeline.VARIANTS)
    assert (table["role"] == "other").all()  # dates outside the configured calendar
    assert np.isfinite(table[["auc", "brier", "ibs_horizon"]]).all().all()
    assert (table["ibs_horizon"] > 0).all()
    for date, rows in table.groupby("snapshot_date"):
        assert rows["observed_rate"].nunique() == 1
        assert (rows["train_rows"] == len(generations.history(date))).all()

    fold = pipeline.forecast(generations, DATES[0], CUTOFF)
    first = table.set_index(["snapshot_date", "model", "variant"]).loc[(DATES[0], "aft", "raw")]
    assert first["predicted_rate"] == pytest.approx(
        1 - fold.surv_by_model["aft_raw"][:, pipeline.K_HORIZON].mean()
    )
    assert first["players"] == len(fold.time)


def test_snapshot_roles_follow_the_configured_calendar():
    roles = {
        config.TUNE_SNAPSHOTS[0]: "tune",
        config.SELECTION_SNAPSHOTS[-1]: "selection",
        config.VAL_SNAPSHOT: "val",
        config.TEST_SNAPSHOT: "test",
        config.TEST_SNAPSHOT + pd.DateOffset(months=1): "later",
        config.VAL_SNAPSHOT + pd.DateOffset(months=1): "other",
    }
    for date, role in roles.items():
        assert pipeline.snapshot_role(date) == role
    # Decisions come first: tuning, then selection, then the calibrator of the test report.
    assert config.TUNE_SNAPSHOTS.max() < config.SELECTION_SNAPSHOTS.min()
    assert config.SELECTION_SNAPSHOTS.max() < config.VAL_SNAPSHOT < config.TEST_SNAPSHOT
    # ...and none of their labels reaches the test snapshot.
    follow_up = pd.Timedelta(days=config.MAX_FOLLOW_UP_DAYS)
    assert config.VAL_SNAPSHOT + follow_up <= config.TEST_SNAPSHOT
    assert config.TEST_SNAPSHOT - config.CALIBRATION_LAG == config.VAL_SNAPSHOT


def _synthetic_fold(seed: int) -> ev.Fold:
    """Exponential durations with curves of known quality under the model names."""
    rng = np.random.default_rng(seed)
    n = 1500
    rate = rng.lognormal(0.0, 0.7, n) / 100
    true_time = np.ceil(rng.exponential(1 / rate))
    event = true_time < config.MAX_FOLLOW_UP_DAYS
    time = np.where(event, true_time, config.MAX_FOLLOW_UP_DAYS)
    true = np.exp(-np.outer(rate, TIMES))
    surv = {
        "cox_raw": true,
        "cox_calibrated": true,  # calibration changes nothing
        "aft_raw": true**4,  # far too pessimistic
        "aft_calibrated": true**2,  # much better, still worse than the truth
    }
    return ev.Fold(np.arange(n), time, event, surv)


def test_select_model_calibrates_only_when_it_helps_and_takes_the_lower_ibs():
    folds = [_synthetic_fold(0), _synthetic_fold(1)]
    summary, candidates = pipeline.select_model(folds, n_boot=40, seed=0)

    assert summary["variants"] == {"cox": "raw", "aft": "calibrated"}
    assert not summary["calibration_effect"]["cox"]["significant"]
    assert summary["calibration_effect"]["aft"]["ci_high"] < 0
    assert (summary["best_model"], summary["best_variant"]) == ("cox", "raw")
    gap = summary["best_vs_runner_up"]
    assert gap["pair"] == "cox_raw - aft_calibrated"
    assert gap["point"] < 0 and gap["ci_high"] < 0 and gap["significant"]

    assert len(candidates) == 4
    selected = candidates[candidates["selected"]]
    assert selected[["model", "variant"]].to_numpy().tolist() == [["cox", "raw"]]
    expected = np.mean(
        [ev.integrated_brier(f.time, f.event, f.surv_by_model["cox_raw"], TIMES) for f in folds]
    )
    assert selected["point"].iloc[0] == pytest.approx(expected)
    assert (candidates["ci_low"] <= candidates["point"]).all()
    assert (candidates["point"] <= candidates["ci_high"]).all()


def test_tune_minimises_the_mean_ibs_over_the_folds(snapshots):
    folds = [pipeline.rolling_fold(snapshots, date, CUTOFF) for date in DATES]
    best, trials = pipeline.tune("aft", folds, n_trials=3, seed=0)

    assert trials["trial"].tolist() == [0, 1, 2]
    assert trials["best_ibs"].iloc[-1] == trials["ibs"].min()
    scores = []
    for history, fold in folds:
        X, time, event = pipeline.xy(fold)
        model = pipeline.make_model("aft", best, seed=0).fit(*pipeline.xy(history))
        scores.append(ev.integrated_brier(time, event, model.predict_survival(X, TIMES), TIMES))
    assert trials["ibs"].min() == pytest.approx(np.mean(scores))


@pytest.mark.parametrize("calibrated", [False, True])
def test_score_players_ranks_by_probability_and_respects_the_delay(
    calibrated, generations, snapshots
):
    date = pd.Timestamp("2024-12-01")
    scoring = snapshots[snapshots["snapshot_date"] == date]
    model = generations.model("cox", date)
    lagged = date - config.CALIBRATION_LAG
    calibrator = generations.calibrator("cox", lagged, date) if calibrated else None
    scores, curves = pipeline.score_players(model, calibrator, scoring)

    probability, raw = f"churn_prob_{HORIZON}d", f"raw_churn_prob_{HORIZON}d"
    assert scores["risk_rank"].tolist() == list(range(1, len(scoring) + 1))
    assert scores[probability].is_monotonic_decreasing
    # Ties of the calibrated probability are ordered by the raw one.
    assert scores.groupby(probability)[raw].apply(lambda tied: tied.is_monotonic_decreasing).all()
    assert calibrated or np.array_equal(scores[probability], scores[raw])
    np.testing.assert_allclose(1 - curves[str(HORIZON)], scores[probability])

    # Churn within 7 days needs at least 24 days of inactivity behind the player.
    cannot_churn_in_a_week = scores["recency_days"] <= config.INACTIVITY_DAYS - 7
    assert (
        cannot_churn_in_a_week.any()
        and (scores.loc[cannot_churn_in_a_week, "churn_prob_7d"] == 0).all()
    )
