"""Tests for metrics, calibration, bootstrap and predicted-vs-observed diagnostics."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn import evaluation as ev

TIMES = np.arange(1, 86)
HORIZON = 60
FOLLOW_UP = 90


@pytest.fixture(scope="module")
def sample() -> dict[str, np.ndarray]:
    """Exponential durations in whole days, administratively censored at day 90."""
    rng = np.random.default_rng(0)
    n = 4000
    rate = rng.lognormal(0.0, 0.7, n) / 100
    true_time = np.ceil(rng.exponential(1 / rate))
    event = true_time < FOLLOW_UP
    time = np.where(event, true_time, FOLLOW_UP)
    return {
        "time": time,
        "event": event,
        "true": np.exp(-np.outer(rate, TIMES)),  # P(T > t) for integer t
        "km": np.tile(ev.km_on_grid(time, event, TIMES), (n, 1)),
    }


def test_km_on_grid_matches_hand_computation():
    time = np.array([1.0, 2.0, 2.0, 3.0])
    event = np.array([True, True, False, True])
    np.testing.assert_allclose(
        ev.km_on_grid(time, event, np.array([0.5, 1, 2, 3])), [1.0, 0.75, 0.5, 0.0]
    )


def test_true_model_beats_uninformative_and_miscalibrated_curves(sample):
    time, event = sample["time"], sample["event"]
    ibs = {
        name: ev.integrated_brier(time, event, surv, TIMES)
        for name, surv in (
            ("true", sample["true"]),
            ("km", sample["km"]),
            ("overconfident", sample["true"] ** 4),
        )
    }
    assert ibs["true"] < ibs["km"] < ibs["overconfident"]

    metrics = ev.summary_metrics(time, event, sample["true"], TIMES, HORIZON)
    assert set(metrics) == set(ev.METRICS)
    assert metrics["auc_horizon"] > 0.65
    assert metrics["c_index"] > 0.6
    assert metrics["brier_horizon"] == pytest.approx(
        ev.brier_over_time(time, event, sample["true"], TIMES)[HORIZON - 1]
    )
    assert not np.isnan(ev.auc_over_time(time, event, sample["true"], TIMES)).any()


def test_metrics_skip_grid_points_outside_follow_up(sample):
    late = sample["time"] >= 5  # no observation before day 5
    time, event, surv = sample["time"][late], sample["event"][late], sample["true"][late]
    brier = ev.brier_over_time(time, event, surv, TIMES)
    assert np.isnan(brier[:4]).all() and not np.isnan(brier[4:]).any()
    assert np.isfinite(ev.integrated_brier(time, event, surv, TIMES))


def test_horizon_metrics_reduce_to_plain_binary_metrics(sample):
    """With every status known at the horizon, IPCW weights are 1."""
    time, event = sample["time"], sample["event"]
    surv_at_horizon = sample["true"][:, HORIZON - 1]
    churned = event & (time <= HORIZON)
    metrics = ev.horizon_metrics(time, event, surv_at_horizon, HORIZON)

    assert metrics["observed_rate"] == pytest.approx(churned.mean())
    assert metrics["predicted_rate"] == pytest.approx(churned.mean(), abs=0.03)
    assert metrics["brier"] == pytest.approx(((1 - surv_at_horizon - churned) ** 2).mean())
    assert metrics["auc"] == pytest.approx(roc_auc_score(churned, 1 - surv_at_horizon))


def test_calibrator_fixes_miscalibrated_curves(sample):
    time, event = sample["time"], sample["event"]
    raw = sample["true"] ** 4  # churn probabilities far too high
    fit, hold = np.arange(len(time)) < 2000, np.arange(len(time)) >= 2000

    calibrator = ev.IsotonicSurvivalCalibrator().fit(raw[fit], time[fit], event[fit], TIMES)
    calibrated = calibrator.transform(raw[hold], TIMES)

    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert np.all(np.diff(calibrated, axis=1) <= 0)
    assert ev.integrated_brier(time[hold], event[hold], calibrated, TIMES) < ev.integrated_brier(
        time[hold], event[hold], raw[hold], TIMES
    )
    observed = (event[hold] & (time[hold] <= HORIZON)).mean()
    assert (1 - calibrated[:, HORIZON - 1]).mean() == pytest.approx(observed, abs=0.03)
    assert (1 - raw[hold][:, HORIZON - 1]).mean() > observed + 0.2

    restored = ev.IsotonicSurvivalCalibrator.from_dict(json.loads(json.dumps(calibrator.to_dict())))
    np.testing.assert_allclose(restored.transform(raw[hold], TIMES), calibrated)


def test_calibrator_guards(sample):
    time, event, surv = sample["time"], sample["event"], sample["true"]
    censored_early = np.where(np.arange(len(time)) == 0, 10.0, time)
    not_event = np.where(np.arange(len(time)) == 0, False, event)
    with pytest.raises(ValueError, match="censored before"):
        ev.IsotonicSurvivalCalibrator().fit(surv, censored_early, not_event, TIMES)

    calibrator = ev.IsotonicSurvivalCalibrator().fit(surv, time, event, TIMES)
    with pytest.raises(ValueError, match="time grid"):
        calibrator.transform(surv[:, :10], TIMES[:10])


def test_calibrator_keeps_certain_survival_certain():
    """Days on which churn is impossible (raw S = 1 exactly) stay at S = 1."""
    rng = np.random.default_rng(1)
    n = 3000
    delay = rng.integers(0, 30, n)
    rate = rng.lognormal(0.0, 0.7, n) / 40
    true_time = delay + np.ceil(rng.exponential(1 / rate))
    event = true_time < FOLLOW_UP
    time = np.where(event, true_time, FOLLOW_UP)
    after_delay = np.clip(TIMES[None, :] - delay[:, None], 0, None)
    raw = np.exp(-rate[:, None] * after_delay) ** 3  # too pessimistic once churn is possible
    fit, hold = np.arange(n) < 1500, np.arange(n) >= 1500

    calibrator = ev.IsotonicSurvivalCalibrator().fit(raw[fit], time[fit], event[fit], TIMES)
    calibrated = calibrator.transform(raw[hold], TIMES)

    impossible = TIMES[None, :] <= delay[hold, None]
    assert impossible.any() and np.all(calibrated[impossible] == 1.0)
    assert ev.integrated_brier(time[hold], event[hold], calibrated, TIMES) < ev.integrated_brier(
        time[hold], event[hold], raw[hold], TIMES
    )


def test_km_by_group_uses_group_curves_and_falls_back_to_the_pooled_one():
    time = np.array([1.0, 2.0, 2.0, 3.0])
    event = np.array([True, True, False, True])
    groups = np.array([5, 5, 7, 7])
    times = np.array([1, 2, 3])
    curves = ev.km_by_group(time, event, groups, np.array([7, 5, 9]), times)
    np.testing.assert_allclose(
        curves,
        [
            [1.0, 1.0, 0.0],  # group 7: censored on day 2, churn on day 3
            [0.5, 0.0, 0.0],  # group 5: churn on days 1 and 2
            ev.km_on_grid(time, event, times),  # unseen group: everybody
        ],
    )


def test_stratified_metrics_match_metrics_of_each_stratum(sample):
    time, event, surv = sample["time"], sample["event"], sample["true"]
    values = np.arange(len(time)) % 10 + 1
    table = ev.stratified_metrics(time, event, surv, TIMES, HORIZON, values, [(1, 3), (4, 10)])

    assert table["stratum"].tolist() == ["1-3", "4-10"]
    assert table["n"].tolist() == [1200, 2800]
    low = values <= 3
    expected = ev.summary_metrics(time[low], event[low], surv[low], TIMES, HORIZON)
    for metric, value in expected.items():
        assert table.loc[0, metric] == pytest.approx(value)
    assert table.loc[0, "churn_rate_horizon"] == pytest.approx(
        (event[low] & (time[low] <= HORIZON)).mean()
    )


def test_bootstrap_mean_ibs_resamples_clusters_across_folds(sample):
    time, event = sample["time"][:1200], sample["event"][:1200]
    surv = {"a": sample["true"][:1200], "b": sample["true"][:1200], "km": sample["km"][:1200]}
    # Two folds of 600 rows; the first 300 clusters are present in both.
    cluster = np.concatenate([np.arange(600), np.arange(300), np.arange(600, 900)])
    folds = [
        ev.Fold(cluster[rows], time[rows], event[rows], {k: s[rows] for k, s in surv.items()})
        for rows in (slice(0, 600), slice(600, 1200))
    ]
    draws = ev.bootstrap_mean_ibs(folds, TIMES, n_boot=20, seed=1)

    assert draws.columns.tolist() == ["draw", "model", "metric", "value"]
    assert len(draws) == 20 * len(surv) and set(draws["metric"]) == {"ibs"}
    point = np.mean(
        [ev.integrated_brier(f.time, f.event, f.surv_by_model["a"], TIMES) for f in folds]
    )
    summary = ev.bootstrap_summary(draws).set_index("model")
    assert summary.loc["a", "ci_low"] <= point <= summary.loc["a", "ci_high"]
    same = ev.bootstrap_difference(draws, "a", "b")
    assert (same[["mean", "ci_low", "ci_high"]] == 0).all().all()
    assert ev.bootstrap_difference(draws, "km", "a")["ci_low"].iloc[0] > 0

    # A cluster is drawn as a whole: with one cluster per fold a draw can only
    # return the fold as it is, so the criterion does not vary.
    whole = [fold._replace(cluster=np.zeros(600, dtype=int)) for fold in folds]
    values = ev.bootstrap_mean_ibs(whole, TIMES, n_boot=5, seed=1).query("model == 'a'")["value"]
    np.testing.assert_allclose(values, point)


def test_bootstrap_summary_and_paired_difference(sample):
    time, event = sample["time"][:800], sample["event"][:800]
    surv = {"a": sample["true"][:800], "b": sample["true"][:800], "km": sample["km"][:800]}
    draws = ev.bootstrap_metrics(time, event, surv, TIMES, HORIZON, n_boot=20, seed=1)
    assert len(draws) == 20 * len(surv) * len(ev.METRICS)

    summary = ev.bootstrap_summary(draws).set_index(["model", "metric"])
    point = ev.summary_metrics(time, event, surv["a"], TIMES, HORIZON)
    for metric, value in point.items():
        assert (
            summary.loc[("a", metric), "ci_low"] <= value <= summary.loc[("a", metric), "ci_high"]
        )

    same = ev.bootstrap_difference(draws, "a", "b")
    assert (same[["mean", "ci_low", "ci_high"]] == 0).all().all()
    worse = ev.bootstrap_difference(draws, "km", "a").set_index("metric")
    assert worse.loc["ibs", "ci_low"] > 0  # the uninformative curve is reliably worse


def test_calibration_table(sample):
    time, event, surv = sample["time"], sample["event"], sample["true"]
    table = ev.calibration_table(time, event, surv, TIMES, HORIZON, n_bins=5)
    assert table["bin"].tolist() == [1, 2, 3, 4, 5]
    assert table["n"].tolist() == [800] * 5
    assert table["predicted"].is_monotonic_increasing
    np.testing.assert_allclose(table["predicted"], table["observed"], atol=0.06)

    with pytest.raises(ValueError, match="censored before"):
        ev.calibration_table(np.full(10, 30.0), np.zeros(10, dtype=bool), surv[:10], TIMES, HORIZON)


def test_hazard_table():
    times = np.arange(1, 4)
    time = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    event = np.array([True, True, False, True, False, False])
    surv = np.tile(np.exp(-0.2 * times), (6, 1))  # constant hazard for everybody
    table = ev.hazard_table(time, event, surv, times, horizon=2)

    assert table["group"].unique().tolist() == ["all", "low", "medium", "high"]
    overall = table[table["group"] == "all"]
    assert overall["at_risk"].tolist() == [6, 5, 3]
    np.testing.assert_allclose(overall["observed"], [1 / 6, 1 / 5, 1 / 3])
    np.testing.assert_allclose(table["predicted"], 1 - np.exp(-0.2))

    with pytest.raises(ValueError, match="daily grid"):
        ev.hazard_table(time, event, surv, times + 1, horizon=2)


def test_expected_days():
    times = np.arange(1, 11)
    churn_on_day_5 = (times < 5).astype(float)[None, :]
    never_churns = np.ones((1, 10))
    assert ev.expected_days(churn_on_day_5, times)[0] == 5
    assert ev.expected_days(never_churns, times)[0] == 10


def test_rul_tables(sample):
    time, event, surv = sample["time"], sample["event"], sample["true"]
    table, summary = ev.rul_tables(time, event, surv, TIMES, sample["km"][0], n_bins=4)
    realised = np.minimum(time, TIMES[-1])

    assert table["n"].sum() == len(time)
    assert table["predicted"].is_monotonic_increasing
    # Without censoring before day 85 the Kaplan-Meier RUL is the plain mean.
    assert np.average(table["observed"], weights=table["n"]) == pytest.approx(realised.mean())
    assert summary["mean_observed"] == pytest.approx(realised.mean())
    assert summary["km_baseline"] == pytest.approx(realised.mean())
    assert summary["mean_predicted"] == pytest.approx(realised.mean(), abs=1.5)
    assert summary["mae_model"] < summary["mae_km"]
    assert summary["rmse_model"] < summary["rmse_km"]
