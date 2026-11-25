"""Survival metrics, calibration, bootstrap and predicted-vs-observed diagnostics.

Every function takes a survival matrix ``surv`` of shape ``(n_samples, len(times))``
with ``surv[i, k] = S(times[k] | x_i)``, so the same code evaluates any model,
raw or calibrated.

The IPCW-based metrics estimate the censoring distribution from the evaluated
sample itself: training rows are censored by the calendar cutoff, which is not
how the evaluation snapshots are censored.
"""

from collections.abc import Sequence
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

METRICS = ("c_index", "ibs", "brier_horizon", "auc_horizon")


def _inside_follow_up(time: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Mask of time points the IPCW metrics accept: ``[min(time), max(time))``."""
    return (times >= time.min()) & (times < time.max())


def _column(times: np.ndarray, horizon: float) -> int:
    """Index of ``horizon`` in ``times``."""
    index = np.flatnonzero(times == horizon)
    if index.size != 1:
        raise ValueError(f"horizon {horizon} is not a point of the time grid")
    return int(index[0])


def require_follow_up(time: np.ndarray, event: np.ndarray, horizon: float) -> None:
    """Check that the churn status at ``horizon`` is known for every sample.

    Raises
    ------
    ValueError
        If a sample is censored before ``horizon``.
    """
    if np.any(~event & (time < horizon)):
        raise ValueError(f"some samples are censored before day {horizon}")


def _require_daily_grid(times: np.ndarray) -> None:
    """Fail unless ``times`` is 1, 2, ..., K, the grid discrete-time formulas assume."""
    if not np.array_equal(times, np.arange(1, len(times) + 1)):
        raise ValueError("times must be the daily grid 1, 2, ..., K")


def _quantile_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    """Equal-sized groups 0..n_groups-1 by increasing value (ties split by position)."""
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, n_groups, labels=False).to_numpy()


def integrated_brier(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, times: np.ndarray
) -> float:
    """Integrated Brier score over the part of ``times`` inside the follow-up.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Increasing time grid.

    Returns
    -------
    float
    """
    y = Surv.from_arrays(event=event, time=time)
    inside = _inside_follow_up(time, times)
    return float(integrated_brier_score(y, y, surv[:, inside], times[inside]))


def brier_over_time(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Time-dependent Brier score at each grid point (NaN outside the follow-up)."""
    y = Surv.from_arrays(event=event, time=time)
    inside = _inside_follow_up(time, times)
    scores = np.full(len(times), np.nan)
    scores[inside] = brier_score(y, y, surv[:, inside], times[inside])[1]
    return scores


def auc_over_time(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Cumulative/dynamic AUC at each grid point (NaN outside the follow-up).

    The risk score at time ``t`` is the predicted churn probability ``1 - S(t | x)``.
    """
    y = Surv.from_arrays(event=event, time=time)
    inside = _inside_follow_up(time, times)
    scores = np.full(len(times), np.nan)
    scores[inside] = cumulative_dynamic_auc(y, y, 1 - surv[:, inside], times[inside])[0]
    return scores


def summary_metrics(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, times: np.ndarray, horizon: float
) -> dict[str, float]:
    """Headline quality metrics of one set of survival curves.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Increasing time grid.
    horizon : float
        Grid point at which the Brier score and AUC are reported; the predicted
        churn probability at this horizon also ranks samples for the C-index.

    Returns
    -------
    dict
        ``c_index`` (Harrell), ``ibs``, ``brier_horizon`` and ``auc_horizon``.
    """
    y = Surv.from_arrays(event=event, time=time)
    k = _column(times, horizon)
    risk = 1 - surv[:, k]
    return {
        "c_index": float(concordance_index_censored(event, time, risk)[0]),
        "ibs": integrated_brier(time, event, surv, times),
        "brier_horizon": float(brier_score(y, y, surv[:, [k]], horizon)[1][0]),
        "auc_horizon": float(cumulative_dynamic_auc(y, y, risk, horizon)[0][0]),
    }


def stratified_metrics(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    times: np.ndarray,
    horizon: float,
    values: np.ndarray,
    strata: Sequence[tuple[float, float]],
) -> pd.DataFrame:
    """:func:`summary_metrics` within strata of one variable.

    The integrated Brier score of a stratum covers only the grid points inside
    the follow-up of that stratum, so it compares models within a stratum, not
    strata with each other.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Increasing time grid.
    horizon : float
        Horizon passed to :func:`summary_metrics`.
    values : np.ndarray
        Stratification variable, one value per sample.
    strata : sequence of (low, high)
        Inclusive bounds of each stratum.

    Returns
    -------
    pd.DataFrame
        One row per stratum: ``stratum`` (``"<low>-<high>"``), ``n``,
        ``churn_rate_horizon`` and the metrics in ``METRICS``.

    Raises
    ------
    ValueError
        If a sample is censored before ``horizon``.
    """
    require_follow_up(time, event, horizon)
    rows = []
    for low, high in strata:
        inside = (values >= low) & (values <= high)
        rows.append(
            {
                "stratum": f"{low}-{high}",
                "n": int(inside.sum()),
                "churn_rate_horizon": float((event[inside] & (time[inside] <= horizon)).mean()),
                **summary_metrics(time[inside], event[inside], surv[inside], times, horizon),
            }
        )
    return pd.DataFrame(rows)


def horizon_metrics(
    time: np.ndarray, event: np.ndarray, surv_at_horizon: np.ndarray, horizon: float
) -> dict[str, float]:
    """Predicted vs observed churn rate at one horizon, with AUC and Brier score.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv_at_horizon : np.ndarray
        Predicted ``S(horizon | x)``, one value per sample.
    horizon : float
        Evaluation horizon; must lie strictly inside the follow-up.

    Returns
    -------
    dict
        ``predicted_rate``, ``observed_rate``, ``auc`` and ``brier``.

    Raises
    ------
    ValueError
        If a sample is censored before ``horizon``.
    """
    require_follow_up(time, event, horizon)
    y = Surv.from_arrays(event=event, time=time)
    churn = 1 - surv_at_horizon
    return {
        "predicted_rate": float(churn.mean()),
        "observed_rate": float((event & (time <= horizon)).mean()),
        "auc": float(cumulative_dynamic_auc(y, y, churn, horizon)[0][0]),
        "brier": float(brier_score(y, y, surv_at_horizon[:, None], horizon)[1][0]),
    }


class IsotonicSurvivalCalibrator:
    """Recalibrate survival curves with one isotonic regression per time point.

    At each grid point ``t`` the predicted churn probability ``1 - S(t | x)`` is
    mapped onto the observed frequency of ``T <= t``; the calibrated curves are
    then made non-increasing in ``t``. The binary target is only defined when
    nobody is censored before the end of the grid, which :meth:`fit` enforces.
    """

    def __init__(self) -> None:
        self.times_: np.ndarray | None = None
        self.knots_: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, surv: np.ndarray, time: np.ndarray, event: np.ndarray, times: np.ndarray) -> Self:
        """Learn the per-time-point mappings.

        Parameters
        ----------
        surv : np.ndarray
            Uncalibrated survival probabilities on a held-out sample.
        time, event : np.ndarray
            Observed durations and boolean event indicators of that sample.
        times : np.ndarray
            Time grid of ``surv``.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If a sample is censored before the last grid point.
        """
        require_follow_up(time, event, times[-1])
        self.times_ = np.asarray(times)
        self.knots_ = []
        for k, t in enumerate(times):
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic.fit(1 - surv[:, k], (event & (time <= t)).astype(float))
            self.knots_.append((isotonic.X_thresholds_, isotonic.y_thresholds_))
        return self

    def transform(self, surv: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calibrate survival probabilities given on the fitted time grid.

        Raises
        ------
        ValueError
            If ``times`` differs from the grid used in :meth:`fit`.
        """
        if self.times_ is None or not np.array_equal(times, self.times_):
            raise ValueError("calibrator must be applied on the time grid it was fitted on")
        churn = np.column_stack(
            [np.interp(1 - surv[:, k], x, y) for k, (x, y) in enumerate(self.knots_)]
        )
        return 1 - np.maximum.accumulate(churn, axis=1)

    def to_dict(self) -> dict[str, list]:
        """JSON-serialisable representation."""
        return {
            "times": self.times_.tolist(),
            "knots": [[x.tolist(), y.tolist()] for x, y in self.knots_],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> Self:
        """Inverse of :meth:`to_dict`."""
        calibrator = cls()
        calibrator.times_ = np.asarray(data["times"])
        calibrator.knots_ = [(np.asarray(x), np.asarray(y)) for x, y in data["knots"]]
        return calibrator


def bootstrap_metrics(
    time: np.ndarray,
    event: np.ndarray,
    surv_by_model: dict[str, np.ndarray],
    times: np.ndarray,
    horizon: float,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Nonparametric bootstrap of :func:`summary_metrics`.

    Rows are resampled with replacement, the same resample being used for every
    model so that differences between models can be bootstrapped as paired.
    Rows must be independent (one row per player).

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv_by_model : dict
        Survival matrices keyed by model name.
    times : np.ndarray
        Time grid of the survival matrices.
    horizon : float
        Horizon passed to :func:`summary_metrics`.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed of the resampling.

    Returns
    -------
    pd.DataFrame
        Long frame with columns ``draw``, ``model``, ``metric``, ``value``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        for model, surv in surv_by_model.items():
            metrics = summary_metrics(time[idx], event[idx], surv[idx], times, horizon)
            rows += [(draw, model, metric, value) for metric, value in metrics.items()]
    return pd.DataFrame(rows, columns=["draw", "model", "metric", "value"])


def bootstrap_summary(draws: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Mean and percentile interval of each bootstrapped metric.

    Returns
    -------
    pd.DataFrame
        Columns ``model``, ``metric``, ``mean``, ``ci_low``, ``ci_high``.
    """
    tail = (1 - level) / 2
    grouped = draws.groupby(["model", "metric"], sort=False)["value"]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "ci_low": grouped.quantile(tail),
            "ci_high": grouped.quantile(1 - tail),
        }
    ).reset_index()


def bootstrap_difference(
    draws: pd.DataFrame, model_a: str, model_b: str, level: float = 0.95
) -> pd.DataFrame:
    """Paired bootstrap of ``metric(model_a) - metric(model_b)``.

    Returns
    -------
    pd.DataFrame
        Columns ``model``, ``metric``, ``mean``, ``ci_low``, ``ci_high`` where
        ``model`` reads ``"<model_a> - <model_b>"``.
    """
    wide = draws.pivot(index=["draw", "metric"], columns="model", values="value")
    difference = (wide[model_a] - wide[model_b]).rename("value").reset_index()
    difference["model"] = f"{model_a} - {model_b}"
    return bootstrap_summary(difference, level)


class Fold(NamedTuple):
    """One evaluation sample with the survival curves of several models."""

    cluster: np.ndarray  # resampling unit of each row, e.g. the player id
    time: np.ndarray
    event: np.ndarray
    surv_by_model: dict[str, np.ndarray]


def bootstrap_mean_ibs(
    folds: Sequence[Fold], times: np.ndarray, n_boot: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Cluster bootstrap of the integrated Brier score averaged over folds.

    Clusters are resampled with replacement and every fold keeps the rows of
    the drawn clusters, so rows of one cluster (a player seen on several
    snapshots) stay together. The same resample is used for every model, which
    makes differences between models paired.

    Parameters
    ----------
    folds : sequence of Fold
        Evaluation samples; every fold holds the same model names.
    times : np.ndarray
        Time grid of the survival matrices.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed of the resampling.

    Returns
    -------
    pd.DataFrame
        Long frame with columns ``draw``, ``model``, ``metric`` (always
        ``"ibs"``) and ``value``, the layout :func:`bootstrap_summary` and
        :func:`bootstrap_difference` take.
    """
    clusters = np.unique(np.concatenate([fold.cluster for fold in folds]))
    positions = [np.searchsorted(clusters, fold.cluster) for fold in folds]
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_boot):
        drawn = rng.integers(0, clusters.size, clusters.size)
        copies = np.bincount(drawn, minlength=clusters.size)
        totals = dict.fromkeys(folds[0].surv_by_model, 0.0)
        for fold, position in zip(folds, positions, strict=True):
            idx = np.repeat(np.arange(len(fold.time)), copies[position])
            for model, surv in fold.surv_by_model.items():
                totals[model] += integrated_brier(fold.time[idx], fold.event[idx], surv[idx], times)
        rows += [(draw, model, "ibs", total / len(folds)) for model, total in totals.items()]
    return pd.DataFrame(rows, columns=["draw", "model", "metric", "value"])


def km_on_grid(time: np.ndarray, event: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival estimate evaluated at ``times``."""
    km_times, km_surv = kaplan_meier_estimator(event, time)
    index = np.searchsorted(km_times, times, side="right")
    return np.concatenate([[1.0], km_surv])[index]


def km_by_group(
    time: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
    new_groups: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Kaplan-Meier curve of each group, handed to new samples by their group.

    A reference that knows one grouping variable and nothing else. Groups absent
    from the reference sample get the pooled Kaplan-Meier curve.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators of the reference sample.
    groups : np.ndarray
        Group of each reference sample.
    new_groups : np.ndarray
        Group of each sample to predict for.
    times : np.ndarray
        Time grid.

    Returns
    -------
    np.ndarray
        Survival probabilities, shape ``(len(new_groups), len(times))``.
    """
    pooled = km_on_grid(time, event, times)
    curves = {
        group: km_on_grid(time[groups == group], event[groups == group], times)
        for group in np.unique(groups)
    }
    return np.vstack([curves.get(group, pooled) for group in new_groups])


def survival_curve_table(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, times: np.ndarray
) -> pd.DataFrame:
    """Mean predicted survival curve next to the Kaplan-Meier estimate.

    Returns
    -------
    pd.DataFrame
        Columns ``day`` (starting at 0), ``predicted``, ``observed``.
    """
    return pd.DataFrame(
        {
            "day": np.concatenate([[0], times]),
            "predicted": np.concatenate([[1.0], surv.mean(axis=0)]),
            "observed": np.concatenate([[1.0], km_on_grid(time, event, times)]),
        }
    )


def calibration_table(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    times: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs observed churn rate by quantile bin of the prediction.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Time grid of ``surv``.
    horizon : float
        Grid point at which churn is evaluated.
    n_bins : int
        Number of equal-sized bins.

    Returns
    -------
    pd.DataFrame
        Columns ``bin`` (1 = lowest risk), ``n``, ``predicted``, ``observed``.

    Raises
    ------
    ValueError
        If a sample is censored before ``horizon``.
    """
    require_follow_up(time, event, horizon)
    predicted = 1 - surv[:, _column(times, horizon)]
    frame = pd.DataFrame(
        {
            "bin": _quantile_groups(predicted, n_bins) + 1,
            "predicted": predicted,
            "observed": event & (time <= horizon),
        }
    )
    grouped = frame.groupby("bin")
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "predicted": grouped["predicted"].mean(),
            "observed": grouped["observed"].mean(),
        }
    ).reset_index()


def hazard_table(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    times: np.ndarray,
    horizon: float,
    groups: tuple[str, ...] = ("low", "medium", "high"),
) -> pd.DataFrame:
    """Predicted vs observed daily churn hazard, overall and by risk group.

    The observed hazard on day ``t`` is ``events on day t / players at risk``.
    The predicted one is the hazard of the group's mean survival curve,
    ``1 - S(t) / S(t - 1)``, which is the quantity the observed ratio estimates
    in a heterogeneous group.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations (whole days) and boolean event indicators.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Daily grid 1, 2, ..., K.
    horizon : float
        Risk groups are equal-sized quantile groups of ``1 - S(horizon | x)``.
    groups : tuple of str
        Group names from lowest to highest risk.

    Returns
    -------
    pd.DataFrame
        Columns ``group`` (``"all"`` plus ``groups``), ``day``, ``at_risk``,
        ``predicted``, ``observed``.
    """
    _require_daily_grid(times)
    risk_group = _quantile_groups(1 - surv[:, _column(times, horizon)], len(groups))
    members = {"all": np.ones(len(time), dtype=bool)}
    members |= {name: risk_group == code for code, name in enumerate(groups)}

    tables = []
    for name, mask in members.items():
        mean_surv = surv[mask].mean(axis=0)
        at_risk = (time[mask, None] >= times).sum(axis=0)
        deaths = (event[mask, None] & (time[mask, None] == times)).sum(axis=0)
        # Undefined (NaN) on days when nobody in the group is left at risk.
        observed = np.divide(deaths, at_risk, out=np.full(len(times), np.nan), where=at_risk > 0)
        tables.append(
            pd.DataFrame(
                {
                    "group": name,
                    "day": times,
                    "at_risk": at_risk,
                    "predicted": 1 - mean_surv / np.concatenate([[1.0], mean_surv[:-1]]),
                    "observed": observed,
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def expected_days(surv: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Remaining useful life: expected days until churn, restricted to the grid.

    For durations in whole days, ``E[min(T, K)] = sum_{t=0}^{K-1} S(t)`` with
    ``S(0) = 1`` and ``K = times[-1]``.

    Parameters
    ----------
    surv : np.ndarray
        Survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Daily grid 1, 2, ..., K.

    Returns
    -------
    np.ndarray
        One value per sample, between 1 and K.
    """
    _require_daily_grid(times)
    return 1.0 + surv[:, :-1].sum(axis=1)


def rul_tables(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    times: np.ndarray,
    baseline_surv: np.ndarray,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare model-based remaining useful life (RUL) with Kaplan-Meier.

    Parameters
    ----------
    time, event : np.ndarray
        Observed durations and boolean event indicators of the evaluated sample.
    surv : np.ndarray
        Predicted survival probabilities, shape ``(n_samples, len(times))``.
    times : np.ndarray
        Daily grid 1, 2, ..., K; RUL is restricted to K days.
    baseline_surv : np.ndarray
        Kaplan-Meier curve of a reference population on ``times``: the single
        RUL one would assign to everybody without a model.
    n_bins : int
        Number of equal-sized bins of predicted RUL.

    Returns
    -------
    table : pd.DataFrame
        Per bin (1 = shortest predicted RUL): ``n``, mean ``predicted`` RUL,
        ``observed`` RUL (Kaplan-Meier within the bin) and ``km_baseline``.
    summary : dict
        Means and the MAE / RMSE of the model and of the Kaplan-Meier baseline
        against the realised ``min(T, K)``.

    Raises
    ------
    ValueError
        If a sample is censored before day K (the realised RUL is then unknown).
    """
    horizon = times[-1]
    require_follow_up(time, event, horizon)
    predicted = expected_days(surv, times)
    baseline = float(expected_days(baseline_surv[None, :], times)[0])
    realised = np.minimum(time, horizon)

    bins = _quantile_groups(predicted, n_bins) + 1
    rows = []
    for b in range(1, n_bins + 1):
        mask = bins == b
        km = km_on_grid(time[mask], event[mask], times)
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "predicted": predicted[mask].mean(),
                "observed": float(expected_days(km[None, :], times)[0]),
                "km_baseline": baseline,
            }
        )
    summary = {
        "restricted_to_days": float(horizon),
        "mean_predicted": float(predicted.mean()),
        "mean_observed": float(realised.mean()),
        "km_baseline": baseline,
        "mae_model": float(np.abs(predicted - realised).mean()),
        "mae_km": float(np.abs(baseline - realised).mean()),
        "rmse_model": float(np.sqrt(((predicted - realised) ** 2).mean())),
        "rmse_km": float(np.sqrt(((baseline - realised) ** 2).mean())),
    }
    return pd.DataFrame(rows), summary
