"""End-to-end pipeline: tune, evaluate, calibrate, backtest, refit, score and export.

Run with ``python -m churn.pipeline`` after ``python -m churn.generate_data``.
Everything the dashboard shows is written to the artifacts directory.
"""

import argparse
import json
import logging
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
SHAP_SAMPLE = 2000  # rows per model kept for importance and dependence plots
FEATURES = list(config.FEATURES)


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split a labelled frame into features, durations and event indicators.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`churn.features.add_labels`.

    Returns
    -------
    X : pd.DataFrame
        The columns in ``config.FEATURES``.
    time, event : np.ndarray
        Durations in days and boolean event indicators.
    """
    return frame[FEATURES], frame["time"].to_numpy(), frame["event"].to_numpy()


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
    stopping, so the production refit (which has no validation set left) can
    reuse it as is.
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
    return MODELS[model_name](params, num_boost_round, seed)


def tune(
    model_name: str, train: pd.DataFrame, val: pd.DataFrame, n_trials: int, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune one model by minimising the integrated Brier score on validation.

    Parameters
    ----------
    model_name : str
        Key of ``churn.models.MODELS``.
    train, val : pd.DataFrame
        Labelled training and validation frames.
    n_trials : int
        Number of Optuna (TPE) trials.
    seed : int
        Seed of the sampler and of the models.

    Returns
    -------
    best_params : dict
        Best hyperparameters, including ``num_boost_round``.
    trials : pd.DataFrame
        Columns ``model``, ``trial``, ``ibs`` and running best ``best_ibs``.
    """
    X_train, t_train, e_train = xy(train)
    X_val, t_val, e_val = xy(val)

    def objective(trial: optuna.Trial) -> float:
        model = make_model(model_name, suggest_params(trial, model_name), seed)
        # A configuration that diverges numerically is discarded, not fatal.
        try:
            surv = model.fit(X_train, t_train, e_train).predict_survival(X_val, TIMES)
        except FloatingPointError as error:
            raise optuna.TrialPruned(str(error)) from error
        if not np.isfinite(surv).all():
            raise optuna.TrialPruned("non-finite survival predictions")
        return ev.integrated_brier(t_val, e_val, surv, TIMES)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    ibs = pd.Series([trial.value for trial in study.trials], dtype=float)
    trials = pd.DataFrame(
        {"model": model_name, "trial": ibs.index, "ibs": ibs, "best_ibs": ibs.cummin()}
    )
    return study.best_params, trials


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
    return importance, pd.concat([sample, shap], axis=1).assign(model=label)


def backtest(
    snapshots: pd.DataFrame, tuned: dict[str, dict[str, Any]], data_cutoff: pd.Timestamp, seed: int
) -> pd.DataFrame:
    """Rolling-origin backtest of the 60-day churn forecast.

    For every backtest date the models are refitted on all earlier snapshots,
    with labels as they were known on that date, and then score the players at
    risk on that date. Hyperparameters are the ones tuned once on the main
    split and predictions are uncalibrated.

    Returns
    -------
    pd.DataFrame
        One row per date and model: sizes, predicted and observed churn rate,
        AUC and Brier score at the horizon.
    """
    rows = []
    for date in config.BACKTEST_SNAPSHOTS:
        history = features.add_labels(snapshots[snapshots["snapshot_date"] < date], date)
        fold = features.add_labels(snapshots[snapshots["snapshot_date"] == date], data_cutoff)
        X, time, event = xy(fold)
        for name in MODELS:
            model = make_model(name, tuned[name], seed).fit(*xy(history))
            surv_at_horizon = model.predict_survival(X, np.array([HORIZON]))[:, 0]
            rows.append(
                {
                    "snapshot_date": date,
                    "model": name,
                    "train_rows": len(history),
                    "players": len(fold),
                    **ev.horizon_metrics(time, event, surv_at_horizon, HORIZON),
                }
            )
    return pd.DataFrame(rows)


def score_players(
    model: SurvivalModel, calibrator: ev.IsotonicSurvivalCalibrator, scoring: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the players at risk on the scoring snapshot, riskiest first.

    Returns
    -------
    scores : pd.DataFrame
        One row per player: features, calibrated churn probabilities at
        ``config.SCORE_HORIZONS``, remaining useful life, risk segment, and the
        three features that push the player's risk up the most.
    survival : pd.DataFrame
        Calibrated survival curves, one column per day of the grid, row-aligned
        with ``scores``.
    """
    X = scoring[FEATURES]
    surv = calibrator.transform(model.predict_survival(X, TIMES), TIMES)
    contributions = model.shap_risk(X)
    top_drivers = np.argsort(-contributions, axis=1)[:, :3]

    scores = scoring[["player_id", *FEATURES]].reset_index(drop=True)
    for h in config.SCORE_HORIZONS:
        scores[f"churn_prob_{h}d"] = 1 - surv[:, TIMES == h][:, 0]
    scores["rul_days"] = ev.expected_days(surv, TIMES)
    scores["risk_score"] = model.predict_risk(X)
    churn_prob = scores[f"churn_prob_{HORIZON}d"]
    names, bounds = zip(*config.RISK_SEGMENTS, strict=True)
    scores["risk_segment"] = np.select([churn_prob >= b for b in bounds], names, default=names[-1])
    for rank in range(3):
        column = top_drivers[:, rank]
        scores[f"driver_{rank + 1}"] = np.array(FEATURES)[column]
        scores[f"driver_{rank + 1}_shap"] = contributions[np.arange(len(X)), column]

    # Calibration creates ties in the probability; the raw risk score breaks them.
    order = scores.sort_values([f"churn_prob_{HORIZON}d", "risk_score"], ascending=False).index
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
        Bootstrap resamples of the test metrics.
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

    # --- Tune on validation IBS, refit, calibrate on validation -------------
    tuned: dict[str, dict[str, Any]] = {}
    evaluation_models: dict[str, SurvivalModel] = {}
    calibrators: dict[str, ev.IsotonicSurvivalCalibrator] = {}
    test_surv: dict[tuple[str, str], np.ndarray] = {}
    quality, trials = [], []
    for name in MODELS:
        tuned[name], model_trials = tune(name, train, val, n_trials, seed)
        trials.append(model_trials)
        model = make_model(name, tuned[name], seed).fit(*xy(train))
        val_raw = model.predict_survival(X_val, TIMES)
        calibrator = ev.IsotonicSurvivalCalibrator().fit(val_raw, t_val, e_val, TIMES)
        test_raw = model.predict_survival(X_test, TIMES)
        test_surv[(name, "raw")] = test_raw
        test_surv[(name, "calibrated")] = calibrator.transform(test_raw, TIMES)
        evaluation_models[name], calibrators[name] = model, calibrator
        quality.append(
            {
                "split": "val",
                "model": name,
                "variant": "raw",
                **ev.summary_metrics(t_val, e_val, val_raw, TIMES, HORIZON),
            }
        )
        logger.info("%s tuned: val IBS=%.5f params=%s", name, quality[-1]["ibs"], tuned[name])
    tables["tuning_trials"] = pd.concat(trials, ignore_index=True)

    # The best model is chosen on validation only; test stays an unbiased report.
    val_ibs = {row["model"]: row["ibs"] for row in quality}
    best = min(val_ibs, key=val_ibs.get)

    # --- Model quality on test (with a no-covariate Kaplan-Meier reference) --
    km_train = ev.km_on_grid(train["time"].to_numpy(), train["event"].to_numpy(), TIMES)
    reference = {("kaplan_meier", "raw"): np.tile(km_train, (len(test), 1))}
    for (name, variant), surv in (test_surv | reference).items():
        metrics = ev.summary_metrics(t_test, e_test, surv, TIMES, HORIZON)
        quality.append({"split": "test", "model": name, "variant": variant, **metrics})
    tables["model_quality"] = pd.DataFrame(quality)
    logger.info("Model quality:\n%s", _show(tables["model_quality"]))
    logger.info("Best model by validation IBS: %s", best)

    # --- Predicted vs observed, hazard, Brier and AUC over time -------------
    tables |= diagnostics(t_test, e_test, test_surv)

    # --- Bootstrap of the test metrics --------------------------------------
    named = {f"{name}_{variant}": surv for (name, variant), surv in test_surv.items()}
    draws = ev.bootstrap_metrics(t_test, e_test, named, TIMES, HORIZON, n_boot, seed)
    point = tables["model_quality"].query("split == 'test' and model != 'kaplan_meier'")
    point = point.assign(model=point["model"] + "_" + point["variant"]).melt(
        id_vars="model", value_vars=list(ev.METRICS), var_name="metric", value_name="point"
    )
    tables["bootstrap_summary"] = ev.bootstrap_summary(draws).merge(point, on=["model", "metric"])
    pairs = [("cox_raw", "aft_raw"), ("cox_calibrated", "aft_calibrated")]
    pairs += [(f"{name}_calibrated", f"{name}_raw") for name in MODELS]
    tables["bootstrap_differences"] = pd.concat(
        [ev.bootstrap_difference(draws, a, b) for a, b in pairs], ignore_index=True
    )
    tables["bootstrap_draws"] = draws

    # --- Remaining useful life: best model vs Kaplan-Meier -------------------
    tables["rul"], rul_summary = ev.rul_tables(
        t_test, e_test, test_surv[(best, "calibrated")], TIMES, km_train
    )
    logger.info("RUL summary: %s", {k: round(v, 3) for k, v in rul_summary.items()})

    # --- Backtest -----------------------------------------------------------
    tables["backtest"] = backtest(snapshots, tuned, data_cutoff, seed)
    logger.info("Backtest:\n%s", _show(tables["backtest"]))

    # --- Production model: best configuration refitted on train + validation -
    production_dates = config.TRAIN_SNAPSHOTS.append(pd.DatetimeIndex([config.VAL_SNAPSHOT]))
    production_train = features.add_labels(
        snapshots[date.isin(production_dates)], config.TEST_SNAPSHOT
    )
    production = make_model(best, tuned[best], seed).fit(*xy(production_train))
    # The production model has seen the validation data, so it reuses the
    # calibrator of the evaluation model; the comparison below checks that.
    production_raw = production.predict_survival(X_test, TIMES)
    compared = {
        ("evaluation", "raw"): test_surv[(best, "raw")],
        ("evaluation", "calibrated"): test_surv[(best, "calibrated")],
        ("production", "raw"): production_raw,
        ("production", "calibrated"): calibrators[best].transform(production_raw, TIMES),
    }
    k = int(np.flatnonzero(TIMES == HORIZON)[0])
    tables["eval_vs_production"] = pd.DataFrame(
        [
            {
                "model": name,
                "variant": variant,
                **ev.summary_metrics(t_test, e_test, surv, TIMES, HORIZON),
                **ev.horizon_metrics(t_test, e_test, surv[:, k], HORIZON),
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
            **{f"{name}_{variant}": 1 - surv[:, k] for (name, variant), surv in compared.items()},
        }
    )
    risk_correlation = spearmanr(
        evaluation_models[best].predict_risk(X_test), production.predict_risk(X_test)
    ).statistic
    churn_prob_gap = np.abs(
        compared[("evaluation", "calibrated")][:, k] - compared[("production", "calibrated")][:, k]
    ).mean()
    agreement = {
        "spearman_risk_score": float(risk_correlation),
        "mean_abs_diff_churn_prob": float(churn_prob_gap),
    }

    # --- Feature importance and SHAP values for dependence plots ------------
    explained = [explain(model, name, X_test, seed) for name, model in evaluation_models.items()]
    explained.append(explain(production, "production", scoring[FEATURES], seed))
    tables["feature_importance"] = pd.concat([imp for imp, _ in explained], ignore_index=True)
    tables["shap_values"] = pd.concat([shap for _, shap in explained], ignore_index=True)

    # --- Scoring of the current player base ---------------------------------
    scores, score_curves = score_players(production, calibrators[best], scoring)
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
    for name, calibrator in calibrators.items():
        (models_dir / f"calibrator_{name}.json").write_text(json.dumps(calibrator.to_dict()))
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "horizon_days": HORIZON,
        "inactivity_days": config.INACTIVITY_DAYS,
        "max_follow_up_days": config.MAX_FOLLOW_UP_DAYS,
        "time_grid": [int(TIMES[0]), int(TIMES[-1])],
        "features": FEATURES,
        "snapshots": {
            "train": [
                str(d.date()) for d in (config.TRAIN_SNAPSHOTS[0], config.TRAIN_SNAPSHOTS[-1])
            ],
            "val": str(config.VAL_SNAPSHOT.date()),
            "test": str(config.TEST_SNAPSHOT.date()),
            "scoring": str(config.SCORING_SNAPSHOT.date()),
        },
        "datasets": {
            "train": describe(train),
            "val": describe(val, full_follow_up=True),
            "test": describe(test, full_follow_up=True),
            "production_train": describe(production_train),
            "scoring": {"rows": len(scoring)},
        },
        "tuning": {"n_trials": n_trials, "val_ibs": val_ibs, "best_params": tuned},
        "best_model": best,
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
