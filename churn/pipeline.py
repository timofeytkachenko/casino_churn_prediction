"""End-to-end pipeline: tune, select, evaluate, calibrate, backtest, refit, score and export.

Run with ``python -m churn.pipeline`` after ``python -m churn.generate_data``.
Everything the dashboard shows is written to the artifacts directory.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr

from churn import config, features
from churn import evaluation as ev
from churn.models import MODELS, SurvivalModel

logger = logging.getLogger(__name__)

TIMES = config.TIME_GRID
HORIZON = config.HORIZON_DAYS
K_HORIZON = int(np.flatnonzero(TIMES == HORIZON)[0])  # column of the horizon in a curve matrix
SHAP_SAMPLE = 2000  # rows per model kept for importance and dependence plots
FEATURES = list(config.FEATURES)
# What a model is given: the features and the delay that sets its time origin.
MODEL_COLUMNS = [*FEATURES, config.DELAY_COLUMN]
VARIANTS = ("raw", "calibrated")


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split a labelled frame into model inputs, durations and event indicators.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`churn.features.add_labels`.

    Returns
    -------
    X : pd.DataFrame
        The columns in ``config.FEATURES`` and ``config.DELAY_COLUMN``.
    time, event : np.ndarray
        Durations in days and boolean event indicators.
    """
    return frame[MODEL_COLUMNS], frame["time"].to_numpy(), frame["event"].to_numpy()


def rolling_fold(
    snapshots: pd.DataFrame, date: pd.Timestamp, label_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training history and evaluation sample of one forecast origin.

    Parameters
    ----------
    snapshots : pd.DataFrame
        Output of :func:`churn.features.build_snapshots`.
    date : pd.Timestamp
        Snapshot date on which the forecast is made.
    label_date : pd.Timestamp
        Date on which the outcomes of the forecast are read.

    Returns
    -------
    history : pd.DataFrame
        Snapshots before ``date`` with labels as they were known on ``date``.
    fold : pd.DataFrame
        Players at risk on ``date`` with labels as known on ``label_date``.
    """
    taken = snapshots["snapshot_date"]
    history = features.add_labels(snapshots[taken < date], date)
    fold = features.add_labels(snapshots[taken == date], label_date)
    return history, fold


def _show(table: pd.DataFrame) -> str:
    """Render a small table for the log."""
    return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def describe(frame: pd.DataFrame, full_follow_up: bool = False) -> dict[str, Any]:
    """Size and outcome summary of a labelled frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`churn.features.add_labels`.
    full_follow_up : bool
        Whether nobody is censored before the horizon; the churn rate at the
        horizon is only reported in that case.

    Returns
    -------
    dict
        ``rows``, ``players``, ``snapshots``, ``event_rate`` and, with
        ``full_follow_up``, ``churn_rate_horizon``.
    """
    summary = {
        "rows": len(frame),
        "players": int(frame["player_id"].nunique()),
        "snapshots": int(frame["snapshot_date"].nunique()),
        "event_rate": float(frame["event"].mean()),
    }
    if full_follow_up:
        churned = frame["event"] & (frame["time"] <= HORIZON)
        summary["churn_rate_horizon"] = float(churned.mean())
    return summary


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Hyperparameter search space shared by both models, plus the AFT noise model.

    The number of boosting rounds is tuned directly instead of using early
    stopping, so every later refit (which has no validation set of its own)
    can reuse it as is.
    """
    params: dict[str, Any] = {
        "num_boost_round": trial.suggest_int("num_boost_round", 50, 500, step=25),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 0.1, 30.0, log=True),
    }
    if model_name == "cox":
        # Cap on the Newton step; kept on because Cox boosting can diverge without it.
        params["max_delta_step"] = trial.suggest_float("max_delta_step", 0.1, 1.0, log=True)
    if model_name == "aft":
        params["aft_loss_distribution"] = trial.suggest_categorical(
            "aft_loss_distribution", ["normal", "logistic", "extreme"]
        )
        params["aft_loss_distribution_scale"] = trial.suggest_float(
            "aft_loss_distribution_scale", 0.2, 5.0, log=True
        )
    return params


def make_model(model_name: str, tuned: dict[str, Any], seed: int) -> SurvivalModel:
    """Instantiate a model from a flat dict of tuned hyperparameters."""
    params = dict(tuned)
    num_boost_round = params.pop("num_boost_round")
    return MODELS[model_name](params, num_boost_round, seed, config.DELAY_COLUMN)


def tune(
    model_name: str,
    folds: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    n_trials: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune one model by minimising the integrated Brier score averaged over folds.

    Parameters
    ----------
    model_name : str
        Key of ``churn.models.MODELS``.
    folds : sequence of (pd.DataFrame, pd.DataFrame)
        Labelled training and validation frames of each fold, e.g. the output
        of :func:`rolling_fold`.
    n_trials : int
        Number of Optuna (TPE) trials.
    seed : int
        Seed of the sampler and of the models.

    Returns
    -------
    best_params : dict
        Best hyperparameters, including ``num_boost_round``.
    trials : pd.DataFrame
        Columns ``model``, ``trial``, ``ibs`` (mean over the folds) and running
        best ``best_ibs``.
    """
    data = [(xy(train), xy(val)) for train, val in folds]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        scores = []
        for (X_train, t_train, e_train), (X_val, t_val, e_val) in data:
            model = make_model(model_name, params, seed)
            # A configuration that diverges numerically is discarded, not fatal.
            try:
                surv = model.fit(X_train, t_train, e_train).predict_survival(X_val, TIMES)
            except FloatingPointError as error:
                raise optuna.TrialPruned(str(error)) from error
            if not np.isfinite(surv).all():
                raise optuna.TrialPruned("non-finite survival predictions")
            scores.append(ev.integrated_brier(t_val, e_val, surv, TIMES))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    ibs = pd.Series([trial.value for trial in study.trials], dtype=float)
    trials = pd.DataFrame(
        {"model": model_name, "trial": ibs.index, "ibs": ibs, "best_ibs": ibs.cummin()}
    )
    return study.best_params, trials


class Generations:
    """Models refitted on everything known on a date, fitted once and reused.

    The generation of date ``d`` is trained on the snapshots before ``d`` with
    labels as they were known on ``d``: the model one would have had that day.
    Its calibrator is fitted on the snapshot of ``d`` itself, which the model
    has not seen, once that snapshot has been followed for the whole time grid.

    Parameters
    ----------
    snapshots : pd.DataFrame
        Output of :func:`churn.features.build_snapshots`.
    tuned : dict
        Hyperparameters by model name, as returned by :func:`tune`.
    seed : int
        Seed of the models.
    """

    def __init__(
        self, snapshots: pd.DataFrame, tuned: dict[str, dict[str, Any]], seed: int
    ) -> None:
        self.snapshots = snapshots
        self.tuned = tuned
        self.seed = seed
        self._histories: dict[pd.Timestamp, pd.DataFrame] = {}
        self._models: dict[tuple[str, pd.Timestamp], SurvivalModel] = {}
        self._calibrators: dict[
            tuple[str, pd.Timestamp, pd.Timestamp], ev.IsotonicSurvivalCalibrator
        ] = {}

    def history(self, date: pd.Timestamp) -> pd.DataFrame:
        """Training data of a generation: earlier snapshots, labels as known on ``date``."""
        if date not in self._histories:
            earlier = self.snapshots[self.snapshots["snapshot_date"] < date]
            self._histories[date] = features.add_labels(earlier, date)
        return self._histories[date]

    def model(self, name: str, date: pd.Timestamp) -> SurvivalModel:
        """Model ``name`` of the generation of ``date``."""
        key = (name, date)
        if key not in self._models:
            model = make_model(name, self.tuned[name], self.seed)
            self._models[key] = model.fit(*xy(self.history(date)))
        return self._models[key]

    def calibrator(
        self, name: str, date: pd.Timestamp, label_date: pd.Timestamp
    ) -> ev.IsotonicSurvivalCalibrator:
        """Calibrator of model ``name`` of the generation of ``date``.

        Parameters
        ----------
        name : str
            Key of ``churn.models.MODELS``.
        date : pd.Timestamp
            Generation; the calibrator is fitted on the players at risk on it.
        label_date : pd.Timestamp
            Date on which the outcomes of those players are read.

        Returns
        -------
        IsotonicSurvivalCalibrator

        Raises
        ------
        ValueError
            If the snapshot has not been followed for the whole grid by ``label_date``.
        """
        key = (name, date, label_date)
        if key not in self._calibrators:
            taken = self.snapshots["snapshot_date"]
            X, time, event = xy(features.add_labels(self.snapshots[taken == date], label_date))
            surv = self.model(name, date).predict_survival(X, TIMES)
            self._calibrators[key] = ev.IsotonicSurvivalCalibrator().fit(surv, time, event, TIMES)
        return self._calibrators[key]


def forecast(
    generations: Generations,
    date: pd.Timestamp,
    label_date: pd.Timestamp,
    lag: pd.DateOffset = config.CALIBRATION_LAG,
) -> ev.Fold:
    """Survival curves of the players at risk on ``date``, as forecast on that day.

    Every model is the generation of ``date``. Its calibrated variant applies
    the calibrator of the generation ``lag`` earlier, the latest one whose
    snapshot has been followed for the whole grid by ``date``. This is the
    procedure that scores the current player base.

    Parameters
    ----------
    generations : Generations
        Source of the refitted models and calibrators.
    date : pd.Timestamp
        Snapshot date of the forecast.
    label_date : pd.Timestamp
        Date on which the outcomes are read.
    lag : pd.DateOffset
        Age of the generation that supplies the calibrator.

    Returns
    -------
    Fold
        Player ids, outcomes and one survival matrix on the time grid per
        ``"<model>_<variant>"``.
    """
    taken = generations.snapshots["snapshot_date"]
    frame = features.add_labels(generations.snapshots[taken == date], label_date)
    X, time, event = xy(frame)
    surv: dict[str, np.ndarray] = {}
    for name in MODELS:
        raw = generations.model(name, date).predict_survival(X, TIMES)
        calibrator = generations.calibrator(name, date - lag, date)
        surv[f"{name}_raw"] = raw
        surv[f"{name}_calibrated"] = calibrator.transform(raw, TIMES)
    return ev.Fold(frame["player_id"].to_numpy(), time, event, surv)


def select_model(
    folds: Sequence[ev.Fold], n_boot: int, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose the model family and whether to calibrate it.

    The criterion is the integrated Brier score averaged over ``folds``. A
    family is represented by its raw curves unless calibration lowers the
    criterion by more than resampling noise, i.e. the 95% interval of
    ``calibrated - raw`` lies below zero. The family with the lower criterion
    wins; the interval of its difference to the runner-up tells whether the
    folds can tell the two apart. Players, not rows, are resampled, because a
    player is at risk on several folds.

    Parameters
    ----------
    folds : sequence of Fold
        Forecasts of every ``"<model>_<variant>"`` on the selection folds, as
        returned by :func:`forecast`.
    n_boot : int
        Bootstrap resamples of the players.
    seed : int
        Seed of the resampling.

    Returns
    -------
    summary : dict
        ``best_model``, ``best_variant``, the variant chosen for every family
        (``variants``), the effect of calibration per family
        (``calibration_effect``) and ``best_vs_runner_up``; effects hold
        ``pair``, ``point``, ``ci_low``, ``ci_high`` and ``significant``.
    candidates : pd.DataFrame
        One row per model and variant: ``point`` (the criterion), bootstrap
        ``mean``, ``ci_low``, ``ci_high`` and ``selected``.
    """
    draws = ev.bootstrap_mean_ibs(folds, TIMES, n_boot, seed)
    point = {
        key: float(
            np.mean(
                [
                    ev.integrated_brier(fold.time, fold.event, fold.surv_by_model[key], TIMES)
                    for fold in folds
                ]
            )
        )
        for key in folds[0].surv_by_model
    }

    def difference(a: str, b: str) -> dict[str, Any]:
        interval = ev.bootstrap_difference(draws, a, b).iloc[0]
        low, high = float(interval["ci_low"]), float(interval["ci_high"])
        return {
            "pair": f"{a} - {b}",
            "point": point[a] - point[b],
            "ci_low": low,
            "ci_high": high,
            "significant": bool(low > 0 or high < 0),
        }

    calibration = {name: difference(f"{name}_calibrated", f"{name}_raw") for name in MODELS}
    variants = {
        name: "calibrated" if effect["ci_high"] < 0 else "raw"
        for name, effect in calibration.items()
    }
    chosen = {name: f"{name}_{variant}" for name, variant in variants.items()}
    best, runner_up = sorted(MODELS, key=lambda name: point[chosen[name]])[:2]
    summary = {
        "criterion": "integrated Brier score averaged over the selection folds",
        "n_bootstrap": n_boot,
        "best_model": best,
        "best_variant": variants[best],
        "variants": variants,
        "calibration_effect": calibration,
        "best_vs_runner_up": difference(chosen[best], chosen[runner_up]),
    }
    candidates = ev.bootstrap_summary(draws)
    candidates["point"] = candidates["model"].map(point)
    candidates["selected"] = candidates["model"] == chosen[best]
    candidates[["model", "variant"]] = candidates["model"].str.rsplit("_", n=1, expand=True)
    columns = ["model", "variant", "point", "mean", "ci_low", "ci_high", "selected"]
    return summary, candidates[columns]


def diagnostics(
    time: np.ndarray, event: np.ndarray, surv_by_variant: dict[tuple[str, str], np.ndarray]
) -> dict[str, pd.DataFrame]:
    """Predicted-vs-observed tables for every (model, variant) on one sample.

    Returns
    -------
    dict
        ``survival_curves``, ``calibration``, ``hazard`` and ``metrics_over_time``.
    """
    tables: dict[str, list[pd.DataFrame]] = {
        "survival_curves": [],
        "calibration": [],
        "hazard": [],
        "metrics_over_time": [],
    }
    for (model, variant), surv in surv_by_variant.items():
        key = {"model": model, "variant": variant}
        tables["survival_curves"].append(
            ev.survival_curve_table(time, event, surv, TIMES).assign(**key)
        )
        tables["hazard"].append(ev.hazard_table(time, event, surv, TIMES, HORIZON).assign(**key))
        tables["metrics_over_time"].append(
            pd.DataFrame(
                {
                    "day": TIMES,
                    "brier": ev.brier_over_time(time, event, surv, TIMES),
                    "auc": ev.auc_over_time(time, event, surv, TIMES),
                }
            ).assign(**key)
        )
        tables["calibration"] += [
            ev.calibration_table(time, event, surv, TIMES, h).assign(horizon=h, **key)
            for h in config.SCORE_HORIZONS
        ]
    return {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}


def explain(
    model: SurvivalModel, label: str, X: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature importance and SHAP values of one model on a sample of rows.

    Both describe the booster, i.e. the time left once churn is possible; the
    delay itself is fixed by the churn definition and has no importance.

    Returns
    -------
    importance : pd.DataFrame
        Columns ``model``, ``feature``, ``gain_share``, ``mean_abs_shap``.
    shap : pd.DataFrame
        Sampled feature values plus one ``shap_<feature>`` column per feature;
        enough to draw a dependence plot for any pair of features.
    """
    sample = X.sample(min(len(X), SHAP_SAMPLE), random_state=seed).reset_index(drop=True)
    contributions = model.shap_risk(sample)
    importance = pd.DataFrame(
        {
            "model": label,
            "feature": FEATURES,
            "gain_share": model.gain_importance().to_numpy(),
            "mean_abs_shap": np.abs(contributions).mean(axis=0),
        }
    )
    shap = pd.DataFrame(contributions, columns=[f"shap_{name}" for name in FEATURES])
    return importance, pd.concat([sample[FEATURES], shap], axis=1).assign(model=label)


def snapshot_role(date: pd.Timestamp) -> str:
    """What the main run uses a snapshot date for.

    Returns
    -------
    str
        ``"tune"``, ``"selection"``, ``"val"`` or ``"test"`` for the dates
        behind a decision or a report, ``"later"`` after the test snapshot and
        ``"other"`` for the remaining dates before it.
    """
    if date in config.TUNE_SNAPSHOTS:
        return "tune"
    if date in config.SELECTION_SNAPSHOTS:
        return "selection"
    if date == config.VAL_SNAPSHOT:
        return "val"
    if date == config.TEST_SNAPSHOT:
        return "test"
    return "later" if date > config.TEST_SNAPSHOT else "other"


def backtest(
    generations: Generations,
    data_cutoff: pd.Timestamp,
    dates: pd.DatetimeIndex = config.BACKTEST_SNAPSHOTS,
    lag: pd.DateOffset = config.CALIBRATION_LAG,
) -> pd.DataFrame:
    """Rolling-origin backtest of the churn forecast, raw and calibrated.

    Every date is forecast by :func:`forecast`: models refitted on all earlier
    snapshots with labels as they were known on that date, calibrated by the
    generation ``lag`` earlier. Hyperparameters are the ones tuned once on the
    tuning folds, so only the folds after them are free of that look-ahead.

    Parameters
    ----------
    generations : Generations
        Source of the refitted models and calibrators.
    data_cutoff : pd.Timestamp
        First day without data; outcomes are read on it.
    dates : pd.DatetimeIndex
        Snapshot dates to forecast.
    lag : pd.DateOffset
        Age of the generation that supplies the calibrator.

    Returns
    -------
    pd.DataFrame
        One row per date, model and variant: ``role`` of the date
        (:func:`snapshot_role`), sizes, predicted and observed churn rate, AUC
        and Brier score at the horizon, and ``ibs_horizon``, the integrated
        Brier score over the days up to the horizon.
    """
    up_to_horizon = TIMES <= HORIZON
    rows = []
    for date in dates:
        fold = forecast(generations, date, data_cutoff, lag)
        for key, surv in fold.surv_by_model.items():
            name, variant = key.rsplit("_", 1)
            rows.append(
                {
                    "snapshot_date": date,
                    "role": snapshot_role(date),
                    "model": name,
                    "variant": variant,
                    "train_rows": len(generations.history(date)),
                    "players": len(fold.time),
                    **ev.horizon_metrics(fold.time, fold.event, surv[:, K_HORIZON], HORIZON),
                    "ibs_horizon": ev.integrated_brier(
                        fold.time, fold.event, surv[:, up_to_horizon], TIMES[up_to_horizon]
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_players(
    model: SurvivalModel,
    calibrator: ev.IsotonicSurvivalCalibrator | None,
    scoring: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the players at risk on the scoring snapshot, riskiest first.

    Parameters
    ----------
    model : SurvivalModel
        Fitted model.
    calibrator : IsotonicSurvivalCalibrator, optional
        Calibrator applied to the curves of ``model``; ``None`` keeps them raw.
    scoring : pd.DataFrame
        Output of :func:`churn.features.snapshot_features`.

    Returns
    -------
    scores : pd.DataFrame
        One row per player: features, churn probabilities at
        ``config.SCORE_HORIZONS``, the uncalibrated probability at the business
        horizon, remaining useful life, risk segment, and the three features
        that push the player's risk up the most.
    survival : pd.DataFrame
        Survival curves behind the scores, one column per day of the grid,
        row-aligned with ``scores``.
    """
    X = scoring[MODEL_COLUMNS]
    raw = model.predict_survival(X, TIMES)
    surv = raw if calibrator is None else calibrator.transform(raw, TIMES)
    contributions = model.shap_risk(X)
    top_drivers = np.argsort(-contributions, axis=1)[:, :3]

    scores = scoring[["player_id", *FEATURES]].reset_index(drop=True)
    for h in config.SCORE_HORIZONS:
        scores[f"churn_prob_{h}d"] = 1 - surv[:, TIMES == h][:, 0]
    scores[f"raw_churn_prob_{HORIZON}d"] = 1 - raw[:, K_HORIZON]
    scores["rul_days"] = ev.expected_days(surv, TIMES)
    scores["risk_score"] = model.predict_risk(X)
    churn_prob = scores[f"churn_prob_{HORIZON}d"]
    names, bounds = zip(*config.RISK_SEGMENTS, strict=True)
    scores["risk_segment"] = np.select([churn_prob >= b for b in bounds], names, default=names[-1])
    for rank in range(3):
        column = top_drivers[:, rank]
        scores[f"driver_{rank + 1}"] = np.array(FEATURES)[column]
        scores[f"driver_{rank + 1}_shap"] = contributions[np.arange(len(X)), column]

    # Calibration creates ties in the probability; the raw probability breaks them.
    # The risk score cannot: it ignores the delay, so it does not order churn by a fixed day.
    ranking = [f"churn_prob_{HORIZON}d", f"raw_churn_prob_{HORIZON}d"]
    order = scores.sort_values(ranking, ascending=False).index
    scores = scores.loc[order].reset_index(drop=True)
    scores.insert(1, "risk_rank", np.arange(1, len(scores) + 1))
    return scores, pd.DataFrame(surv[order], columns=[str(t) for t in TIMES])


def run(data_dir: Path, artifacts_dir: Path, n_trials: int, n_boot: int, seed: int) -> None:
    """Run the whole pipeline and write the artifacts.

    Parameters
    ----------
    data_dir : Path
        Directory with ``players.parquet`` and ``activity.parquet``.
    artifacts_dir : Path
        Output directory (created if missing; existing files are overwritten).
    n_trials : int
        Optuna trials per model.
    n_boot : int
        Bootstrap resamples of the selection criterion and of the test metrics.
    seed : int
        Seed for tuning, model fitting, sampling and the bootstrap.
    """
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    # --- Data: one landmark dataset per split -------------------------------
    players, activity = features.load_raw(data_dir)
    snapshots = features.build_snapshots(players, activity, config.ALL_SNAPSHOTS)
    data_cutoff = config.DATA_END + pd.Timedelta(days=1)
    date = snapshots["snapshot_date"]
    train = features.add_labels(snapshots[date.isin(config.TRAIN_SNAPSHOTS)], config.VAL_SNAPSHOT)
    val = features.add_labels(snapshots[date == config.VAL_SNAPSHOT], config.TEST_SNAPSHOT)
    test = features.add_labels(snapshots[date == config.TEST_SNAPSHOT], data_cutoff)
    scoring = snapshots[date == config.SCORING_SNAPSHOT]
    X_val, t_val, e_val = xy(val)
    X_test, t_test, e_test = xy(test)
    logger.info("Rows: train=%d val=%d test=%d scoring=%d", *map(len, (train, val, test, scoring)))

    # --- Tune on the early rolling-origin folds ------------------------------
    # Every decision before the test snapshot reads its labels on that date at the latest.
    tune_folds = [rolling_fold(snapshots, d, config.TEST_SNAPSHOT) for d in config.TUNE_SNAPSHOTS]
    tuned: dict[str, dict[str, Any]] = {}
    trials = []
    for name in MODELS:
        tuned[name], model_trials = tune(name, tune_folds, n_trials, seed)
        trials.append(model_trials)
        logger.info(
            "%s tuned: mean IBS over the tuning folds=%.5f params=%s",
            name,
            model_trials["best_ibs"].iloc[-1],
            tuned[name],
        )
    tables["tuning_trials"] = pd.concat(trials, ignore_index=True)
    tune_ibs = tables["tuning_trials"].groupby("model", sort=False)["ibs"].min().to_dict()
    generations = Generations(snapshots, tuned, seed)

    # --- Choose the family and whether to calibrate, on the selection folds --
    selection_folds = [
        forecast(generations, d, config.TEST_SNAPSHOT) for d in config.SELECTION_SNAPSHOTS
    ]
    selection, tables["selection"] = select_model(selection_folds, n_boot, seed)
    best, best_variant = selection["best_model"], selection["best_variant"]
    logger.info("Selection:\n%s", _show(tables["selection"]))
    logger.info("Selected: %s %s; %s", best, best_variant, selection["best_vs_runner_up"])

    # --- Evaluation models: trained on train, calibrated on validation -------
    # The generation of the validation date is exactly the model fitted on ``train``,
    # and its calibrator is fitted on ``val``.
    evaluation_models: dict[str, SurvivalModel] = {}
    calibrators: dict[str, ev.IsotonicSurvivalCalibrator] = {}
    test_surv: dict[tuple[str, str], np.ndarray] = {}
    quality = []
    for name in MODELS:
        model = generations.model(name, config.VAL_SNAPSHOT)
        calibrator = generations.calibrator(name, config.VAL_SNAPSHOT, config.TEST_SNAPSHOT)
        test_raw = model.predict_survival(X_test, TIMES)
        test_surv[(name, "raw")] = test_raw
        test_surv[(name, "calibrated")] = calibrator.transform(test_raw, TIMES)
        evaluation_models[name], calibrators[name] = model, calibrator
        val_raw = model.predict_survival(X_val, TIMES)
        metrics = ev.summary_metrics(t_val, e_val, val_raw, TIMES, HORIZON)
        quality.append({"split": "val", "model": name, "variant": "raw", **metrics})
    val_ibs = {row["model"]: row["ibs"] for row in quality}

    # --- Model quality on test, next to two references without a model -------
    t_train, e_train = train["time"].to_numpy(), train["event"].to_numpy()
    km_train = ev.km_on_grid(t_train, e_train, TIMES)
    recency_train, recency_test = (f["recency_days"].to_numpy() for f in (train, test))
    km_recency = ev.km_by_group(t_train, e_train, recency_train, recency_test, TIMES)
    reference = {
        ("kaplan_meier", "raw"): np.tile(km_train, (len(test), 1)),
        ("km_recency", "raw"): km_recency,
    }
    for (name, variant), surv in (test_surv | reference).items():
        metrics = ev.summary_metrics(t_test, e_test, surv, TIMES, HORIZON)
        quality.append({"split": "test", "model": name, "variant": variant, **metrics})
    tables["model_quality"] = pd.DataFrame(quality)
    logger.info("Model quality:\n%s", _show(tables["model_quality"]))

    # Pooled metrics are dominated by long-inactive players; look inside the strata too.
    parts = [
        ev.stratified_metrics(t_test, e_test, surv, TIMES, HORIZON, recency_test, [bounds]).assign(
            model=name, variant=variant
        )
        for bounds in config.RECENCY_STRATA
        for (name, variant), surv in (test_surv | {("km_recency", "raw"): km_recency}).items()
    ]
    strata = pd.concat(parts, ignore_index=True)
    leading = ["stratum", "model", "variant"]
    tables["quality_by_recency"] = strata[leading + [c for c in strata if c not in leading]]
    logger.info("Quality by recency:\n%s", _show(tables["quality_by_recency"]))

    # --- Predicted vs observed, hazard, Brier and AUC over time -------------
    tables |= diagnostics(t_test, e_test, test_surv)

    # --- Bootstrap of the test metrics --------------------------------------
    named = {f"{name}_{variant}": surv for (name, variant), surv in test_surv.items()}
    named["km_recency_raw"] = km_recency
    draws = ev.bootstrap_metrics(t_test, e_test, named, TIMES, HORIZON, n_boot, seed)
    point = tables["model_quality"].query("split == 'test' and model != 'kaplan_meier'")
    point = point.assign(model=point["model"] + "_" + point["variant"]).melt(
        id_vars="model", value_vars=list(ev.METRICS), var_name="metric", value_name="point"
    )
    tables["bootstrap_summary"] = ev.bootstrap_summary(draws).merge(point, on=["model", "metric"])
    pairs = [("cox_raw", "aft_raw"), ("cox_calibrated", "aft_calibrated")]
    pairs += [(f"{name}_calibrated", f"{name}_raw") for name in MODELS]
    pairs += [(key, "km_recency_raw") for key in named if key != "km_recency_raw"]
    tables["bootstrap_differences"] = pd.concat(
        [ev.bootstrap_difference(draws, a, b) for a, b in pairs], ignore_index=True
    )
    tables["bootstrap_draws"] = draws

    # --- Remaining useful life: selected model vs Kaplan-Meier ---------------
    tables["rul"], rul_summary = ev.rul_tables(
        t_test, e_test, test_surv[(best, best_variant)], TIMES, km_train
    )
    logger.info("RUL summary: %s", {k: round(v, 3) for k, v in rul_summary.items()})

    # --- Backtest -----------------------------------------------------------
    tables["backtest"] = backtest(generations, data_cutoff)
    logger.info("Backtest:\n%s", _show(tables["backtest"]))

    # --- Production: the selected family refitted on everything known today --
    production = generations.model(best, config.SCORING_SNAPSHOT)
    production_train = generations.history(config.SCORING_SNAPSHOT)
    # Its calibrator comes from the generation one lag earlier, whose snapshot has
    # just been followed for the whole grid; with the default lag that is the test
    # snapshot, which is why it is fitted only after the test report above.
    calibration_date = config.SCORING_SNAPSHOT - config.CALIBRATION_LAG
    production_calibrator = generations.calibrator(best, calibration_date, config.SCORING_SNAPSHOT)

    # The same procedure replayed on the test date: what production would have
    # predicted there, next to the evaluation model frozen one lag earlier.
    replay = forecast(generations, config.TEST_SNAPSHOT, data_cutoff)
    compared = {
        ("evaluation", "raw"): test_surv[(best, "raw")],
        ("evaluation", "calibrated"): test_surv[(best, "calibrated")],
        ("production", "raw"): replay.surv_by_model[f"{best}_raw"],
        ("production", "calibrated"): replay.surv_by_model[f"{best}_calibrated"],
    }
    tables["eval_vs_production"] = pd.DataFrame(
        [
            {
                "model": name,
                "variant": variant,
                **ev.summary_metrics(t_test, e_test, surv, TIMES, HORIZON),
                **ev.horizon_metrics(t_test, e_test, surv[:, K_HORIZON], HORIZON),
            }
            for (name, variant), surv in compared.items()
        ]
    ).drop(columns=["auc", "brier"])  # already reported as auc_horizon / brier_horizon
    logger.info("Evaluation vs production:\n%s", _show(tables["eval_vs_production"]))
    tables["eval_vs_production_curves"] = pd.concat(
        [
            ev.survival_curve_table(t_test, e_test, surv, TIMES).assign(model=name, variant=variant)
            for (name, variant), surv in compared.items()
        ],
        ignore_index=True,
    )
    tables["eval_vs_production_scores"] = pd.DataFrame(
        {
            "player_id": test["player_id"].to_numpy(),
            "churned": e_test & (t_test <= HORIZON),
            **{
                f"{name}_{variant}": 1 - surv[:, K_HORIZON]
                for (name, variant), surv in compared.items()
            },
        }
    )
    risk_correlation = spearmanr(
        evaluation_models[best].predict_risk(X_test),
        generations.model(best, config.TEST_SNAPSHOT).predict_risk(X_test),
    ).statistic
    churn_prob_gap = np.abs(
        compared[("evaluation", best_variant)][:, K_HORIZON]
        - compared[("production", best_variant)][:, K_HORIZON]
    ).mean()
    agreement = {
        "spearman_risk_score": float(risk_correlation),
        "mean_abs_diff_churn_prob": float(churn_prob_gap),
    }

    # --- Feature importance and SHAP values for dependence plots ------------
    explained = [explain(model, name, X_test, seed) for name, model in evaluation_models.items()]
    explained.append(explain(production, "production", scoring[MODEL_COLUMNS], seed))
    tables["feature_importance"] = pd.concat([imp for imp, _ in explained], ignore_index=True)
    tables["shap_values"] = pd.concat([shap for _, shap in explained], ignore_index=True)

    # --- Scoring of the current player base ---------------------------------
    scoring_calibrator = production_calibrator if best_variant == "calibrated" else None
    scores, score_curves = score_players(production, scoring_calibrator, scoring)
    score_curves.insert(0, "player_id", scores["player_id"])
    tables["scores"] = scores
    tables["score_curves"] = score_curves
    tables["top_risk_users"] = scores.head(config.TOP_N_RISKY)
    shown = ["risk_rank", "player_id", "recency_days", f"churn_prob_{HORIZON}d", "rul_days"]
    logger.info("Top 10 risky players:\n%s", _show(scores.head(10)[shown]))

    # --- Artifacts ----------------------------------------------------------
    for name, table in tables.items():
        table.to_parquet(artifacts_dir / f"{name}.parquet", index=False)
    for name, model in (evaluation_models | {"production": production}).items():
        model.save(models_dir / name)
    for name, calibrator in (calibrators | {"production": production_calibrator}).items():
        (models_dir / f"calibrator_{name}.json").write_text(json.dumps(calibrator.to_dict()))
    production_snapshots = production_train["snapshot_date"]
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "horizon_days": HORIZON,
        "inactivity_days": config.INACTIVITY_DAYS,
        "max_follow_up_days": config.MAX_FOLLOW_UP_DAYS,
        "calibration_lag_months": config.CALIBRATION_LAG.months,
        "time_grid": [int(TIMES[0]), int(TIMES[-1])],
        "features": FEATURES,
        "snapshots": {
            "train": [
                str(d.date()) for d in (config.TRAIN_SNAPSHOTS[0], config.TRAIN_SNAPSHOTS[-1])
            ],
            "val": str(config.VAL_SNAPSHOT.date()),
            "test": str(config.TEST_SNAPSHOT.date()),
            "scoring": str(config.SCORING_SNAPSHOT.date()),
            "tune": [str(d.date()) for d in config.TUNE_SNAPSHOTS],
            "selection": [str(d.date()) for d in config.SELECTION_SNAPSHOTS],
            "production_train": [
                str(d.date()) for d in (production_snapshots.min(), production_snapshots.max())
            ],
            "production_calibration": str(calibration_date.date()),
        },
        "datasets": {
            "train": describe(train),
            "val": describe(val, full_follow_up=True),
            "test": describe(test, full_follow_up=True),
            "production_train": describe(production_train),
            "scoring": {"rows": len(scoring)},
        },
        "tuning": {
            "n_trials": n_trials,
            "tune_ibs": tune_ibs,
            "val_ibs": val_ibs,
            "best_params": tuned,
        },
        "selection": selection,
        "best_model": best,
        "best_variant": best_variant,
        "n_bootstrap": n_boot,
        "rul": rul_summary,
        "eval_vs_production_agreement": agreement,
        "risk_segments": dict(config.RISK_SEGMENTS),
        "seed": seed,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Artifacts written to %s", artifacts_dir)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--n-trials", type=int, default=40, help="Optuna trials per model")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args.data_dir, args.artifacts_dir, args.n_trials, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
