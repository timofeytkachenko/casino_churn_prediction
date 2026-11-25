"""Streamlit dashboard over the pipeline artifacts (English by default, Russian available).

Run with ``streamlit run dashboard/app.py`` after ``python -m churn.pipeline``.
The app only reads ``artifacts/``; it never imports the modelling code.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from i18n import LANGUAGES, Translator
from plotly.subplots import make_subplots

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
GREY = "#6b7280"
COLORS = {
    "cox": "#2563eb",
    "aft": "#f97316",
    "production": "#16a34a",
    "kaplan_meier": "#9ca3af",
    "km_recency": "#a855f7",
    "all": GREY,
    "low": "#16a34a",
    "medium": "#f59e0b",
    "high": "#dc2626",
}
DASHES = {"raw": "dot", "calibrated": "solid"}
METRICS = ("ibs", "brier_horizon", "auc_horizon", "c_index")


# --- Helpers ------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, modified: float) -> pd.DataFrame:
    """Read a table; ``modified`` makes the cache follow pipeline re-runs."""
    return pd.read_parquet(path)


def table(name: str) -> pd.DataFrame:
    """Load one artifact table."""
    path = ARTIFACTS / f"{name}.parquet"
    return _read_parquet(str(path), path.stat().st_mtime)


def series_name(t: Translator, model: str, variant: str) -> str:
    """Legend label of a (model, variant) pair."""
    return f"{t.label(model)} · {t.label(variant)}"


def key_name(t: Translator, key: str) -> str:
    """Legend label of a ``<model>_<variant>`` key."""
    return series_name(t, *key.rsplit("_", 1))


def add_line(
    fig: go.Figure, x, y, name: str, color: str, dash: str = "solid", width: float = 2, **kwargs
) -> None:
    """Add a line trace."""
    line = {"color": color, "dash": dash, "width": width}
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=line, **kwargs))


def show_chart(
    fig: go.Figure, x_title: str | None, y_title: str | None, height: int = 430, **layout
) -> None:
    """Apply the common layout and render; ``None`` keeps an axis title as it is."""
    titles = {"xaxis_title": x_title, "yaxis_title": y_title}
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        **{name: title for name, title in titles.items() if title is not None},
        **layout,
    )
    st.plotly_chart(fig, width="stretch")


def show_table(t: Translator, frame: pd.DataFrame, percent: tuple[str, ...] = ()) -> None:
    """Render a table with translated headers and labels."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.date
        elif not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].astype(str).map(t.label)
    formats = {t.label(column): "{:.1%}" for column in percent}
    styled = out.rename(columns=t.label).style.format(formats, precision=4, na_rep="—")
    st.dataframe(styled, hide_index=True, width="stretch")


def show_differences(t: Translator, differences: pd.DataFrame) -> None:
    """Paired bootstrap differences with a flag for intervals that exclude zero."""
    out = differences.rename(columns={"model": "difference"})
    out["difference"] = [
        " − ".join(key_name(t, key) for key in pair.split(" - ")) for pair in out["difference"]
    ]
    excludes_zero = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out["significant"] = np.where(excludes_zero, t("yes"), t("no"))
    show_table(t, out)


def pick_variant(
    t: Translator, frame: pd.DataFrame, key: str, default: tuple[str, str]
) -> tuple[str, str]:
    """Select box over the (model, variant) pairs present in ``frame``."""
    pairs = frame[["model", "variant"]].drop_duplicates()
    options = list(pairs.itertuples(index=False, name=None))
    return st.selectbox(
        t("model_variant"),
        options,
        index=options.index(default),
        format_func=lambda pair: series_name(t, *pair),
        key=key,
    )


def plot_positions(values: pd.Series) -> tuple[np.ndarray, list[str] | None]:
    """Numeric plotting positions and, for categorical data, the category names."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return values.cat.codes.to_numpy(dtype=float), list(values.cat.categories)
    return values.to_numpy(dtype=float), None


# --- Sections -----------------------------------------------------------------


def render_overview(t: Translator, metrics: dict) -> None:
    best, variant = metrics["best_model"], metrics["best_variant"]
    quality, scores = table("model_quality"), table("scores")
    row = quality.query("split == 'test' and model == @best and variant == @variant").iloc[0]
    high_risk = (scores["risk_segment"] == "high").sum()
    kpis = st.columns(5)
    kpis[0].metric(t("kpi_best_model"), t.label(best))
    kpis[1].metric(t("kpi_test_ibs", variant=t.label(variant)), f"{row['ibs']:.4f}")
    kpis[2].metric(t("kpi_test_auc"), f"{row['auc_horizon']:.3f}")
    kpis[3].metric(t("kpi_churn_rate"), f"{metrics['datasets']['test']['churn_rate_horizon']:.1%}")
    kpis[4].metric(t("kpi_high_risk"), f"{high_risk:,} / {len(scores):,}")
    st.markdown(
        t(
            "setup_text",
            inactivity=metrics["inactivity_days"],
            horizon=metrics["horizon_days"],
            follow_up=metrics["max_follow_up_days"],
            grid_end=metrics["time_grid"][1],
        )
    )

    st.subheader(t("datasets"))
    snapshots = metrics["snapshots"]
    spans = {
        "train": " .. ".join(snapshots["train"]),
        "val": snapshots["val"],
        "test": snapshots["test"],
        "production_train": " .. ".join(snapshots["production_train"]),
        "scoring": snapshots["scoring"],
    }
    datasets = pd.DataFrame(
        [
            {"split": name, "snapshot_dates": spans[name], **stats}
            for name, stats in metrics["datasets"].items()
        ]
    ).drop(columns="snapshots")
    datasets["players"] = datasets["players"].astype("Int64")  # unknown for the scoring set
    show_table(t, datasets, percent=("event_rate", "churn_rate_horizon"))

    st.subheader(t("tuning"))
    folds = ", ".join(snapshots["tune"])
    st.caption(t("tuning_note", n_trials=metrics["tuning"]["n_trials"], folds=folds))
    left, right = st.columns([2, 3])
    params = pd.DataFrame(metrics["tuning"]["best_params"]).map(
        lambda v: "—" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else str(v)
    )
    params = params.rename(columns=t.label).rename_axis(t("hyperparameter")).reset_index()
    left.dataframe(params, hide_index=True, width="stretch")
    fig = go.Figure()
    for model, part in table("tuning_trials").groupby("model", sort=False):
        name = t.label(model)
        fig.add_trace(
            go.Scatter(
                x=part["trial"],
                y=part["ibs"],
                mode="markers",
                name=f"{name} · {t('trial_value')}",
                marker={"color": COLORS[model], "opacity": 0.35},
            )
        )
        add_line(
            fig, part["trial"], part["best_ibs"], f"{name} · {t('best_so_far')}", COLORS[model]
        )
    with right:
        show_chart(fig, t("trial"), t.label("ibs"), height=360, yaxis_type="log")


def render_quality(t: Translator, metrics: dict) -> None:
    quality, selection = table("model_quality"), metrics["selection"]
    st.success(
        t(
            "best_model_text",
            model=t.label(selection["best_model"]),
            variant=t.label(selection["best_variant"]),
            folds=", ".join(metrics["snapshots"]["selection"]),
        )
    )
    st.subheader(t("selection_title"))
    candidates = table("selection")
    show_table(t, candidates.assign(selected=np.where(candidates["selected"], t("yes"), t("no"))))
    effects = [*selection["calibration_effect"].values(), selection["best_vs_runner_up"]]
    differences = pd.DataFrame(effects).rename(columns={"pair": "model"})
    show_differences(t, differences[["model", "point", "ci_low", "ci_high"]])
    st.caption(t("selection_note", n_boot=selection["n_bootstrap"]))

    st.subheader(t("quality_title"))
    show_table(t, quality)
    st.caption(t("quality_note", horizon=metrics["horizon_days"]))

    metric = st.selectbox(t("metric"), METRICS, format_func=t.label, key="quality_metric")
    test = quality[quality["split"] == "test"]
    fig = go.Figure(
        go.Bar(
            x=[series_name(t, m, v) for m, v in zip(test["model"], test["variant"], strict=True)],
            y=test[metric],
            marker_color=[COLORS[m] for m in test["model"]],
            text=test[metric].round(4),
        )
    )
    show_chart(fig, "", t.label(metric), height=360)

    st.subheader(t("strata_title"))
    st.caption(t("strata_note"))
    show_table(t, table("quality_by_recency"), percent=("churn_rate_horizon",))


def render_pred_obs(t: Translator, metrics: dict) -> None:
    st.subheader(t("survival_title"))
    st.caption(t("survival_note"))
    curves = table("survival_curves")
    observed = curves.drop_duplicates("day")
    fig = go.Figure()
    add_line(fig, observed["day"], observed["observed"], t("observed_km"), GREY, width=5)
    for (model, variant), part in curves.groupby(["model", "variant"], sort=False):
        name = series_name(t, model, variant)
        add_line(fig, part["day"], part["predicted"], name, COLORS[model], DASHES[variant])
    show_chart(fig, t("day"), t("survival_probability"), yaxis_tickformat=".0%")

    st.subheader(t("deciles_title"))
    calibration = table("calibration")
    left, right = st.columns(2)
    with left:
        model, variant = pick_variant(
            t, calibration, "pred_obs_variant", (metrics["best_model"], metrics["best_variant"])
        )
    horizons = sorted(calibration["horizon"].unique())
    horizon = right.selectbox(t("horizon"), horizons, index=len(horizons) - 1, key="pred_obs_h")
    part = calibration.query("model == @model and variant == @variant and horizon == @horizon")
    fig = go.Figure(
        [
            go.Bar(
                x=part["bin"], y=part["predicted"], name=t("predicted"), marker_color=COLORS[model]
            ),
            go.Bar(x=part["bin"], y=part["observed"], name=t("observed"), marker_color=GREY),
        ]
    )
    show_chart(fig, t("risk_decile"), t("churn_rate_by", horizon=horizon), yaxis_tickformat=".0%")


def render_hazard(t: Translator, metrics: dict) -> None:
    hazard = table("hazard")
    st.caption(
        t("hazard_note", horizon=metrics["horizon_days"], inactivity=metrics["inactivity_days"])
    )
    left, middle, right = st.columns([2, 2, 1])
    with left:
        model, variant = pick_variant(
            t, hazard, "hazard_variant", (metrics["best_model"], metrics["best_variant"])
        )
    groups = middle.multiselect(
        t("risk_groups"),
        ["all", "low", "medium", "high"],
        default=["all", "high"],
        format_func=t.label,
    )
    window = right.slider(t("smoothing_days"), 1, 14, 7)
    part = hazard.query("model == @model and variant == @variant")
    fig = go.Figure()
    for group in groups:
        rows = part[part["group"] == group]
        smooth = rows[["predicted", "observed"]].rolling(window, center=True, min_periods=1).mean()
        name, color = t.label(group), COLORS[group]
        add_line(fig, rows["day"], smooth["predicted"], f"{name} · {t('predicted')}", color)
        add_line(fig, rows["day"], smooth["observed"], f"{name} · {t('observed')}", color, "dot")
    show_chart(fig, t("day"), t("daily_hazard"), yaxis_tickformat=".1%")


def render_over_time(t: Translator, metrics: dict) -> None:
    over_time = table("metrics_over_time")
    st.caption(t("over_time_note"))
    for metric in ("brier", "auc"):
        st.subheader(t(f"{metric}_title"))
        fig = go.Figure()
        for (model, variant), part in over_time.groupby(["model", "variant"], sort=False):
            name = series_name(t, model, variant)
            add_line(fig, part["day"], part[metric], name, COLORS[model], DASHES[variant])
        fig.add_vline(x=metrics["horizon_days"], line_dash="dash", line_color=GREY)
        show_chart(fig, t("day"), t.label(metric), height=360)


def render_calibration(t: Translator, metrics: dict) -> None:
    calibration = table("calibration")
    st.caption(t("calibration_note", grid_end=metrics["time_grid"][1]))
    horizons = sorted(calibration["horizon"].unique())
    horizon = st.selectbox(t("horizon"), horizons, index=len(horizons) - 1, key="calibration_h")
    part = calibration[calibration["horizon"] == horizon]
    models = list(part["model"].unique())
    titles = [t.label(model) for model in models]
    fig = make_subplots(cols=len(models), subplot_titles=titles, shared_yaxes=True)
    for col, model in enumerate(models, start=1):
        diagonal = go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", showlegend=False, line={"color": GREY, "dash": "dash"}
        )
        fig.add_trace(diagonal, row=1, col=col)
        for variant, rows in part[part["model"] == model].groupby("variant", sort=False):
            trace = go.Scatter(
                x=rows["predicted"],
                y=rows["observed"],
                mode="lines+markers",
                name=series_name(t, model, variant),
                line={"color": COLORS[model], "dash": DASHES[variant]},
            )
            fig.add_trace(trace, row=1, col=col)
        fig.update_xaxes(title_text=t("predicted_churn_rate"), tickformat=".0%", row=1, col=col)
    show_chart(fig, None, t("observed_churn_rate"), yaxis_tickformat=".0%")

    st.subheader(t("calibration_effect"))
    quality = table("model_quality")
    quality = quality[(quality["split"] == "test") & quality["model"].isin(models)]
    show_table(t, quality.drop(columns="split"))
    differences = table("bootstrap_differences")
    own = [f"{model}_calibrated - {model}_raw" for model in models]
    show_differences(t, differences[differences["model"].isin(own)])


def render_bootstrap(t: Translator, metrics: dict) -> None:
    summary = table("bootstrap_summary")
    st.caption(t("bootstrap_note", n_boot=metrics["n_bootstrap"]))
    metric = st.selectbox(t("metric"), METRICS, format_func=t.label, key="bootstrap_metric")
    part = summary[summary["metric"] == metric]
    fig = go.Figure(
        go.Scatter(
            x=part["point"],
            y=[key_name(t, key) for key in part["model"]],
            mode="markers",
            marker={"size": 12, "color": [COLORS[key.rsplit("_", 1)[0]] for key in part["model"]]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": part["ci_high"] - part["point"],
                "arrayminus": part["point"] - part["ci_low"],
            },
        )
    )
    show_chart(fig, t.label(metric), "", height=320)
    show_table(t, summary.assign(model=[key_name(t, key) for key in summary["model"]]))

    st.subheader(t("paired_differences"))
    st.caption(t("paired_note"))
    show_differences(t, table("bootstrap_differences"))


def render_importance(t: Translator, metrics: dict) -> None:
    importance = table("feature_importance")
    st.caption(t("importance_note"))
    kind = st.radio(
        t("importance_type"), ["mean_abs_shap", "gain_share"], format_func=t.label, horizontal=True
    )
    # Shares within a model make Cox (log hazard) and AFT (log time) comparable.
    importance["share"] = importance[kind] / importance.groupby("model")[kind].transform("sum")
    order = importance.groupby("feature")["share"].mean().sort_values().index
    fig = go.Figure()
    for model, part in importance.groupby("model", sort=False):
        fig.add_trace(
            go.Bar(
                y=[t.label(feature) for feature in order],
                x=part.set_index("feature").loc[order, "share"],
                name=t.label(model),
                orientation="h",
                marker_color=COLORS[model],
            )
        )
    show_chart(fig, t("importance_share"), "", height=560, xaxis_tickformat=".0%", barmode="group")
    with st.expander(t("raw_values")):
        show_table(t, importance.drop(columns="share"))


def render_dependence(t: Translator, metrics: dict) -> None:
    shap, features = table("shap_values"), metrics["features"]
    st.caption(t("dependence_note", n_pairs=len(features) * (len(features) - 1)))
    left, middle, right = st.columns(3)
    model = left.selectbox(t("model"), list(shap["model"].unique()), format_func=t.label)
    feature = middle.selectbox(
        t("feature"), features, index=features.index("recency_days"), format_func=t.label
    )
    others = [name for name in features if name != feature]
    color_feature = right.selectbox(t("color_feature"), others, format_func=t.label)
    show_all = st.checkbox(t("all_interactions"))

    part = shap[shap["model"] == model]
    x, categories = plot_positions(part[feature])
    x_axis = {}
    if categories:
        # Spread the points of each category so that they do not hide each other.
        x = x + np.random.default_rng(0).uniform(-0.25, 0.25, len(x))
        ticks = list(range(len(categories)))
        x_axis = {"tickvals": ticks, "ticktext": [t.label(name) for name in categories]}

    def scatter(colored_by: str, with_scale: bool) -> go.Scattergl:
        values, names = plot_positions(part[colored_by])
        # Clip the colour range so that a few extreme players do not flatten it.
        low, high = (0, len(names) - 1) if names else np.nanpercentile(values, [5, 95])
        colorbar = {"title": {"text": t.label(colored_by), "side": "right"}}
        if names:
            colorbar |= {
                "tickvals": list(range(len(names))),
                "ticktext": [t.label(name) for name in names],
            }
        marker = {
            "size": 5,
            "opacity": 0.6,
            "color": values,
            "colorscale": "Viridis",
            "cmin": low,
            "cmax": high,
            "showscale": with_scale,
            "colorbar": colorbar,
        }
        return go.Scattergl(x=x, y=part[f"shap_{feature}"], mode="markers", marker=marker)

    if show_all:
        st.caption(t("all_interactions_note", feature=t.label(feature)))
        n_cols = 3
        n_rows = -(-len(others) // n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[t.label(name) for name in others],
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
        )
        for position, name in enumerate(others):
            fig.add_trace(
                scatter(name, False), row=position // n_cols + 1, col=position % n_cols + 1
            )
        fig.update_xaxes(**x_axis)
        show_chart(fig, None, None, height=250 * n_rows, showlegend=False)
    else:
        fig = go.Figure(scatter(color_feature, True))
        fig.update_xaxes(**x_axis)
        y_title = t("shap_axis", feature=t.label(feature))
        show_chart(fig, t.label(feature), y_title, height=520)


def render_rul(t: Translator, metrics: dict) -> None:
    rul, summary, best = table("rul"), metrics["rul"], metrics["best_model"]
    days = int(summary["restricted_to_days"])
    variant = t.label(metrics["best_variant"])
    st.caption(t("rul_note", model=t.label(best), variant=variant, days=days))
    kpis = st.columns(4)
    kpis[0].metric(t("rul_mean_predicted"), f"{summary['mean_predicted']:.1f}")
    kpis[1].metric(t("rul_mean_observed"), f"{summary['mean_observed']:.1f}")
    kpis[2].metric(
        t("rul_mae_model"),
        f"{summary['mae_model']:.1f}",
        delta=f"{summary['mae_model'] - summary['mae_km']:.1f} {t('vs_km')}",
        delta_color="inverse",
    )
    kpis[3].metric(t("rul_mae_km"), f"{summary['mae_km']:.1f}")
    fig = go.Figure(
        [
            go.Bar(
                x=rul["bin"], y=rul["predicted"], name=t("rul_model"), marker_color=COLORS[best]
            ),
            go.Bar(x=rul["bin"], y=rul["observed"], name=t("rul_observed_km"), marker_color=GREY),
        ]
    )
    add_line(fig, rul["bin"], rul["km_baseline"], t("rul_km_baseline"), COLORS["high"], "dash")
    show_chart(fig, t("rul_decile"), t("rul_axis", days=days))
    show_table(t, rul)


def render_backtest(t: Translator, metrics: dict) -> None:
    backtest, horizon = table("backtest"), metrics["horizon_days"]
    lag = t("calibration_lag", months=metrics["calibration_lag_months"])
    st.caption(t("backtest_note", horizon=horizon, lag=lag))
    observed = backtest.drop_duplicates("snapshot_date")
    fig = go.Figure()
    add_line(
        fig, observed["snapshot_date"], observed["observed_rate"], t("observed"), GREY, width=5
    )
    series = [
        (series_name(t, model, variant), COLORS[model], DASHES[variant], part)
        for (model, variant), part in backtest.groupby(["model", "variant"], sort=False)
    ]
    for name, color, dash, part in series:
        add_line(fig, part["snapshot_date"], part["predicted_rate"], name, color, dash)
    y_title = t("churn_rate_by", horizon=horizon)
    show_chart(fig, t("snapshot_date"), y_title, yaxis_tickformat=".0%", yaxis_rangemode="tozero")
    for column, metric in zip(st.columns(3), ("auc", "brier", "ibs_horizon"), strict=True):
        fig = go.Figure()
        for name, color, dash, part in series:
            add_line(fig, part["snapshot_date"], part[metric], name, color, dash)
        with column:
            show_chart(fig, t("snapshot_date"), t.label(metric), height=320)
    show_table(t, backtest, percent=("predicted_rate", "observed_rate"))


def render_production(t: Translator, metrics: dict) -> None:
    datasets, agreement = metrics["datasets"], metrics["eval_vs_production_agreement"]
    best, variant, horizon = metrics["best_model"], metrics["best_variant"], metrics["horizon_days"]
    snapshots = metrics["snapshots"]
    st.caption(
        t(
            "production_note",
            model=t.label(best),
            eval_rows=datasets["train"]["rows"],
            prod_rows=datasets["production_train"]["rows"],
            scoring=snapshots["scoring"],
            calibration=snapshots["production_calibration"],
            val=snapshots["val"],
            test=snapshots["test"],
        )
    )
    kpis = st.columns(2)
    kpis[0].metric(t("agreement_rank"), f"{agreement['spearman_risk_score']:.4f}")
    kpis[1].metric(t("agreement_gap"), f"{agreement['mean_abs_diff_churn_prob']:.4f}")
    show_table(t, table("eval_vs_production"), percent=("predicted_rate", "observed_rate"))

    colors = {"evaluation": COLORS[best], "production": COLORS["production"]}
    curves = table("eval_vs_production_curves")
    observed = curves.drop_duplicates("day")
    fig = go.Figure()
    add_line(fig, observed["day"], observed["observed"], t("observed_km"), GREY, width=5)
    for (model, variant), part in curves.groupby(["model", "variant"], sort=False):
        name = series_name(t, model, variant)
        add_line(fig, part["day"], part["predicted"], name, colors[model], DASHES[variant])
    left, right = st.columns(2)
    with left:
        st.subheader(t("survival_title"))
        show_chart(fig, t("day"), t("survival_probability"), yaxis_tickformat=".0%")

    scores = table("eval_vs_production_scores")
    points = go.Scattergl(
        x=scores[f"evaluation_{variant}"],
        y=scores[f"production_{variant}"],
        mode="markers",
        marker={"size": 4, "opacity": 0.4, "color": COLORS["production"]},
        showlegend=False,
    )
    fig = go.Figure(points)
    add_line(fig, [0, 1], [0, 1], "y = x", GREY, "dash", showlegend=False)
    with right:
        st.subheader(t("score_agreement_title", horizon=horizon, variant=t.label(variant)))
        show_chart(fig, t.label("evaluation"), t.label("production"))


def render_risk(t: Translator, metrics: dict) -> None:
    scores, curves = table("scores"), table("score_curves")
    horizon, inactivity = metrics["horizon_days"], metrics["inactivity_days"]
    segments = [name for name in ("high", "medium", "low") if name in metrics["risk_segments"]]
    variant = t.label(metrics["best_variant"])
    date = metrics["snapshots"]["scoring"]
    st.caption(t("risk_note", date=date, variant=variant, inactivity=inactivity))

    left, right = st.columns(2)
    counts = scores["risk_segment"].value_counts().reindex(segments, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[t.label(name) for name in segments],
            y=counts.to_numpy(),
            marker_color=[COLORS[name] for name in segments],
            text=counts.to_numpy(),
        )
    )
    with left:
        show_chart(fig, t("risk_segment"), t("players"), height=300)
    histogram = go.Histogram(
        x=scores[f"churn_prob_{horizon}d"],
        xbins={"start": 0, "end": 1.0001, "size": 0.025},
        marker_color=COLORS[metrics["best_model"]],
    )
    with right:
        x_title = t("churn_prob_axis", horizon=horizon)
        show_chart(go.Figure(histogram), x_title, t("players"), height=300, xaxis_tickformat=".0%")

    st.subheader(t("top_title"))
    first, second, third = st.columns(3)
    top_n = first.slider(t("top_n"), 10, 200, 50, step=10)
    max_recency = second.slider(t("max_recency"), 1, inactivity, inactivity)
    chosen = third.multiselect(t("risk_segment"), segments, default=segments, format_func=t.label)
    eligible = (scores["recency_days"] <= max_recency) & scores["risk_segment"].isin(chosen)
    top = scores[eligible].head(top_n)  # scores are stored riskiest first
    if top.empty:
        st.info(t("no_players"))
        return
    probabilities = [column for column in scores.columns if column.startswith("churn_prob_")]
    drivers = ["driver_1", "driver_2", "driver_3"]
    shown = ["risk_rank", "player_id", *probabilities, "rul_days", "risk_segment"]
    shown += ["recency_days", "active_days_30d", *drivers]
    show_table(t, top[shown], percent=tuple(probabilities))

    st.subheader(t("scorecard_title"))
    player_id = st.selectbox(t("player"), top["player_id"])
    player = top[top["player_id"] == player_id].iloc[0]
    kpis = st.columns(len(probabilities) + 2)
    for column, name in zip(kpis, probabilities, strict=False):
        days = name.removeprefix("churn_prob_").removesuffix("d")
        column.metric(t("churn_within", days=days), f"{player[name]:.1%}")
    kpis[-2].metric(t.label("rul_days"), f"{player['rul_days']:.1f}")
    kpis[-1].metric(t("risk_segment"), t.label(player["risk_segment"]))

    survival = curves.set_index("player_id")
    days = [0, *survival.columns.astype(int)]
    fig = go.Figure()
    own_curve = [1.0, *survival.loc[player_id]]
    add_line(fig, days, own_curve, t("player_curve"), COLORS[player["risk_segment"]], width=3)
    add_line(fig, days, [1.0, *survival.mean()], t("average_curve"), GREY, "dash")
    fig.add_vline(x=horizon, line_dash="dot", line_color=GREY)
    left, right = st.columns([3, 2])
    with left:
        layout = {"yaxis_tickformat": ".0%", "yaxis_range": [0, 1.02]}
        show_chart(fig, t("day"), t("survival_probability"), height=360, **layout)
    with right:
        st.markdown(f"**{t('drivers_title')}**")
        driver_table = pd.DataFrame(
            {
                "feature": [player[name] for name in drivers],
                "value": [str(player[player[name]]) for name in drivers],
                "shap": [player[f"{name}_shap"] for name in drivers],
            }
        )
        show_table(t, driver_table)
        st.caption(t("drivers_note"))
    with st.expander(t("all_features")):
        profile = player[metrics["features"]].astype(str)
        show_table(t, profile.rename_axis("feature").reset_index(name="value"))


SECTIONS = {
    "overview": render_overview,
    "quality": render_quality,
    "pred_obs": render_pred_obs,
    "hazard": render_hazard,
    "over_time": render_over_time,
    "calibration": render_calibration,
    "bootstrap": render_bootstrap,
    "importance": render_importance,
    "dependence": render_dependence,
    "rul": render_rul,
    "backtest": render_backtest,
    "production": render_production,
    "risk": render_risk,
}


def main() -> None:
    """Page chrome: language switch, section navigation, then the chosen section."""
    st.set_page_config(page_title="Casino churn survival", layout="wide")
    language = st.sidebar.radio("Language / Язык", list(LANGUAGES), horizontal=True)
    t = Translator(LANGUAGES[language])

    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        st.error(t("missing_artifacts", path=ARTIFACTS))
        st.stop()
    metrics = json.loads(metrics_path.read_text())

    section = st.sidebar.radio(
        t("section"), list(SECTIONS), format_func=lambda key: t(f"section_{key}")
    )
    st.sidebar.divider()
    st.sidebar.caption(
        t(
            "sidebar_info",
            best=series_name(t, metrics["best_model"], metrics["best_variant"]),
            scoring=metrics["snapshots"]["scoring"],
            generated=metrics["generated_at"][:16].replace("T", " "),
        )
    )
    st.title(t("app_title", horizon=metrics["horizon_days"]))
    st.header(t(f"section_{section}"))
    SECTIONS[section](t, metrics)


main()
