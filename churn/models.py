"""XGBoost survival models (Cox and AFT) behind one survival-curve interface."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, ndtr

_BASE_PARAMS = {"tree_method": "hist", "verbosity": 0}


def breslow_baseline(
    time: np.ndarray, event: np.ndarray, log_hazard: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimate of the baseline cumulative hazard of a Cox model.

    Parameters
    ----------
    time : np.ndarray
        Observed durations.
    event : np.ndarray
        Boolean event indicators (False = right-censored).
    log_hazard : np.ndarray
        Fitted log hazard ratios, one per sample.

    Returns
    -------
    event_times : np.ndarray
        Sorted distinct event times.
    cumulative_hazard : np.ndarray
        Baseline cumulative hazard right after each of ``event_times``.
    """
    order = np.argsort(time, kind="stable")
    time, event = time[order], event[order]
    hazard_ratio = np.exp(log_hazard[order])
    # at_risk[k] = sum of hazard ratios over samples with time >= k-th distinct time.
    distinct, first = np.unique(time, return_index=True)
    at_risk = np.cumsum(hazard_ratio[::-1])[::-1][first]
    deaths = np.bincount(np.searchsorted(distinct, time[event]), minlength=distinct.size)
    has_event = deaths > 0
    return distinct[has_event], np.cumsum(deaths[has_event] / at_risk[has_event])


class SurvivalModel(ABC):
    """Gradient-boosted survival model that can output survival curves.

    Parameters
    ----------
    params : dict, optional
        XGBoost booster parameters (learning rate, depth, regularisation, ...).
    num_boost_round : int
        Number of boosting rounds.
    seed : int
        Seed for row and column subsampling.
    """

    name: ClassVar[str]
    objective: ClassVar[str]
    # +1 if a larger raw margin means earlier churn, -1 if it means later churn.
    risk_sign: ClassVar[int]
    # Objective-specific booster defaults; ``params`` takes precedence.
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(
        self, params: dict[str, Any] | None = None, num_boost_round: int = 200, seed: int = 0
    ) -> None:
        self.params = dict(params or {})
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.booster_: xgb.Booster | None = None
        self.feature_names_: list[str] = []
        # Fitted state beyond the booster, kept JSON-serialisable for save/load.
        self.state_: dict[str, list[float]] = {}

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> Self:
        """Train on right-censored durations.

        Parameters
        ----------
        X : pd.DataFrame
            Features; categorical columns must have the ``category`` dtype.
        time : np.ndarray
            Strictly positive durations.
        event : np.ndarray
            Boolean event indicators (False = right-censored).

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If any duration is not strictly positive.
        """
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        if np.any(time <= 0):
            raise ValueError("durations must be strictly positive")
        self.feature_names_ = list(X.columns)
        params = {**_BASE_PARAMS, **self.default_params, **self.params}
        params |= {"objective": self.objective, "seed": self.seed}
        matrix = self._training_matrix(X, time, event)
        self.booster_ = xgb.train(params, matrix, self.num_boost_round)
        return self

    @abstractmethod
    def _training_matrix(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> xgb.DMatrix:
        """Encode censored durations the way the objective expects them."""

    def _matrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        if self.booster_ is None:
            raise RuntimeError("model is not fitted")
        return xgb.DMatrix(X[self.feature_names_], enable_categorical=True)

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        """Raw booster output: log hazard ratio (Cox) or mean log time (AFT)."""
        return self.booster_.predict(self._matrix(X), output_margin=True).astype(float)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Risk score on the margin scale; larger means earlier churn."""
        return self.risk_sign * self.predict_margin(X)

    @abstractmethod
    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Survival probabilities ``S(t | x)``.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        times : np.ndarray
            Strictly positive time points.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(X), len(times))``.
        """

    def shap_risk(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP contributions to the risk score (positive = pushes towards churn).

        Returns
        -------
        np.ndarray
            Array of shape ``(len(X), n_features)``; the bias term is dropped.
        """
        contributions = self.booster_.predict(self._matrix(X), pred_contribs=True)
        return self.risk_sign * contributions[:, :-1].astype(float)

    def gain_importance(self) -> pd.Series:
        """Share of the total split gain attributed to each feature."""
        gain = self.booster_.get_score(importance_type="total_gain")
        total = pd.Series(gain, dtype=float).reindex(self.feature_names_, fill_value=0.0)
        return total / total.sum()

    def save(self, prefix: Path) -> None:
        """Write ``<prefix>.booster.json`` and ``<prefix>.meta.json``."""
        self.booster_.save_model(f"{prefix}.booster.json")
        meta = {
            "model": self.name,
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "seed": self.seed,
            "feature_names": self.feature_names_,
            "state": self.state_,
        }
        Path(f"{prefix}.meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, prefix: Path) -> Self:
        """Restore a model written by :meth:`save`.

        Raises
        ------
        ValueError
            If the files were written by a different model class.
        """
        meta = json.loads(Path(f"{prefix}.meta.json").read_text())
        if meta["model"] != cls.name:
            raise ValueError(f"{prefix} holds a {meta['model']!r} model, not {cls.name!r}")
        model = cls(meta["params"], meta["num_boost_round"], meta["seed"])
        model.booster_ = xgb.Booster(model_file=f"{prefix}.booster.json")
        model.feature_names_ = meta["feature_names"]
        model.state_ = meta["state"]
        return model


class CoxModel(SurvivalModel):
    """XGBoost ``survival:cox`` with a Breslow baseline hazard.

    ``S(t | x) = exp(-H0(t) * exp(eta(x)))`` where ``eta`` is the booster margin
    and ``H0`` is estimated on the training data after boosting.

    Near-certain events (a player one day short of the inactivity threshold)
    push ``eta`` towards infinity while the hessian of the partial likelihood
    vanishes, so an unconstrained Newton step can overflow. ``max_delta_step``
    caps the step, the same safeguard XGBoost applies to Poisson regression.
    """

    name = "cox"
    objective = "survival:cox"
    risk_sign = 1
    default_params: ClassVar[dict[str, Any]] = {"max_delta_step": 0.7}

    def _training_matrix(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> xgb.DMatrix:
        # XGBoost's Cox objective marks right-censored rows with a negative label.
        return xgb.DMatrix(X, label=np.where(event, time, -time), enable_categorical=True)

    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> Self:
        """Train the booster, then estimate the baseline hazard.

        Raises
        ------
        ValueError
            If any duration is not strictly positive.
        FloatingPointError
            If boosting diverged to non-finite log hazard ratios.
        """
        super().fit(X, time, event)
        log_hazard = self.predict_margin(X)
        if not np.isfinite(log_hazard).all():
            raise FloatingPointError(
                "Cox boosting diverged to non-finite margins; lower max_delta_step "
                "or regularise more"
            )
        times, hazard = breslow_baseline(
            np.asarray(time, dtype=float), np.asarray(event, dtype=bool), log_hazard
        )
        self.state_ = {"baseline_times": times.tolist(), "baseline_hazard": hazard.tolist()}
        return self

    def baseline_cumulative_hazard(self, times: np.ndarray) -> np.ndarray:
        """Right-continuous step function ``H0(t)``, zero before the first event."""
        index = np.searchsorted(self.state_["baseline_times"], times, side="right")
        return np.concatenate([[0.0], self.state_["baseline_hazard"]])[index]

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        hazard_ratio = np.exp(self.predict_margin(X))
        return np.exp(-np.outer(hazard_ratio, self.baseline_cumulative_hazard(np.asarray(times))))


class AFTModel(SurvivalModel):
    """XGBoost ``survival:aft``.

    ``log T = mu(x) + sigma * Z`` with ``Z`` standard normal, logistic or
    extreme-value, so ``S(t | x) = P(Z > (log t - mu(x)) / sigma)``.
    """

    name = "aft"
    objective = "survival:aft"
    risk_sign = -1
    # Survival function of the standardised noise Z for each XGBoost distribution.
    _noise_survival: ClassVar[dict[str, Any]] = {
        "normal": lambda z: ndtr(-z),
        "logistic": lambda z: expit(-z),
        "extreme": lambda z: np.exp(-np.exp(z)),
    }

    @property
    def distribution(self) -> str:
        """Noise distribution (XGBoost default: normal)."""
        return self.params.get("aft_loss_distribution", "normal")

    @property
    def scale(self) -> float:
        """Noise scale ``sigma`` (XGBoost default: 1)."""
        return float(self.params.get("aft_loss_distribution_scale", 1.0))

    def _training_matrix(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> xgb.DMatrix:
        matrix = xgb.DMatrix(X, enable_categorical=True)
        matrix.set_float_info("label_lower_bound", time)
        matrix.set_float_info("label_upper_bound", np.where(event, time, np.inf))
        return matrix

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if np.any(times <= 0):
            raise ValueError("times must be strictly positive")
        z = (np.log(times)[None, :] - self.predict_margin(X)[:, None]) / self.scale
        return self._noise_survival[self.distribution](z)


MODELS: dict[str, type[SurvivalModel]] = {CoxModel.name: CoxModel, AFTModel.name: AFTModel}
