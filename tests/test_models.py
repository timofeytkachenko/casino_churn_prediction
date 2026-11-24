"""Tests for the XGBoost Cox and AFT survival models."""

import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator

from churn.models import MODELS, AFTModel, CoxModel, breslow_baseline

TIMES = np.arange(1, 61)
PARAMS = {"learning_rate": 0.1, "max_depth": 3}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Weibull durations (both a PH and an AFT model) with random censoring."""
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "segment": pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"]),
        }
    )
    log_time = (
        3.0 + 0.8 * X["x1"] - 0.5 * (X["segment"] == "c") + 0.5 * np.log(rng.exponential(1, n))
    )
    true_time = np.ceil(np.exp(log_time))
    censor_time = np.ceil(rng.uniform(5, 120, n))
    event = (true_time <= censor_time).to_numpy()
    return X, np.minimum(true_time, censor_time).to_numpy(), event


@pytest.fixture(scope="module", params=sorted(MODELS))
def fitted(request, data):
    X, time, event = data
    return MODELS[request.param](PARAMS, num_boost_round=60).fit(X, time, event)


def test_breslow_with_unit_hazard_ratios_is_nelson_aalen(data):
    _, time, event = data
    times, cumulative_hazard = breslow_baseline(time, event, np.zeros_like(time))
    expected_times, expected = nelson_aalen_estimator(event, time)
    at_events = np.isin(expected_times, times)
    np.testing.assert_array_equal(times, expected_times[at_events])
    np.testing.assert_allclose(cumulative_hazard, expected[at_events])


def test_cox_cumulative_hazards_sum_to_number_of_events(data):
    """Breslow identity: fitted cumulative hazards at the observed times add up to the events."""
    X, time, event = data
    model = CoxModel(PARAMS, num_boost_round=60).fit(X, time, event)
    own_time_hazard = np.exp(model.predict_margin(X)) * model.baseline_cumulative_hazard(time)
    assert own_time_hazard.sum() == pytest.approx(event.sum(), rel=1e-6)


def test_survival_curves_are_valid_and_ordered_by_risk(fitted, data):
    X, time, event = data
    survival = fitted.predict_survival(X, TIMES)
    risk = fitted.predict_risk(X)

    assert survival.shape == (len(X), TIMES.size)
    assert np.all((survival >= 0) & (survival <= 1))
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    # A higher risk score means a lower survival probability at every horizon.
    order = np.argsort(risk)
    assert np.all(np.diff(survival[order], axis=0) <= 1e-9)
    assert concordance_index_censored(event, time, risk)[0] > 0.7


@pytest.mark.parametrize(
    ("distribution", "survival_at_location"),
    [("normal", 0.5), ("logistic", 0.5), ("extreme", np.exp(-1))],
)
def test_aft_survival_matches_closed_form(data, distribution, survival_at_location):
    X, time, event = data
    params = {**PARAMS, "aft_loss_distribution": distribution, "aft_loss_distribution_scale": 0.7}
    model = AFTModel(params, num_boost_round=20).fit(X.head(300), time[:300], event[:300])
    row = X.head(1)
    location = np.exp(model.predict_margin(row))  # where the standardised noise is zero
    assert model.predict_survival(row, location)[0, 0] == pytest.approx(survival_at_location)


def test_shap_contributions_add_up_to_the_risk_score(fitted, data):
    X, _, _ = data
    contributions = fitted.shap_risk(X)
    assert contributions.shape == (len(X), X.shape[1])
    bias = fitted.predict_risk(X) - contributions.sum(axis=1)
    assert np.ptp(bias) < 1e-3
    assert fitted.gain_importance().sum() == pytest.approx(1.0)


def test_save_load_round_trip(fitted, data, tmp_path):
    X, _, _ = data
    fitted.save(tmp_path / "model")
    restored = type(fitted).load(tmp_path / "model")
    np.testing.assert_allclose(
        restored.predict_survival(X, TIMES), fitted.predict_survival(X, TIMES), rtol=1e-6
    )
    other = AFTModel if isinstance(fitted, CoxModel) else CoxModel
    with pytest.raises(ValueError, match="holds a"):
        other.load(tmp_path / "model")


def test_fit_rejects_non_positive_durations(data):
    X, time, event = data
    with pytest.raises(ValueError, match="strictly positive"):
        CoxModel(PARAMS).fit(X, np.where(np.arange(len(time)) == 0, 0.0, time), event)
