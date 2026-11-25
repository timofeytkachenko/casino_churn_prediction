"""English and Russian strings for the dashboard."""

LANGUAGES = {"English": "en", "Русский": "ru"}

# key -> (English, Russian). Keys are either UI message ids or raw data values
# (model names, column names, feature names) that need a readable label.
TEXT: dict[str, tuple[str, str]] = {
    # --- Chrome ---------------------------------------------------------------
    "app_title": (
        "Online casino churn: {horizon}-day survival forecast",
        "Отток игроков онлайн-казино: прогноз на {horizon} дней",
    ),
    "section": ("Section", "Раздел"),
    "sidebar_info": (
        "Best model: {best}  \nScoring snapshot: {scoring}  \nArtifacts built: {generated} UTC",
        "Лучшая модель: {best}  \nСрез скоринга: {scoring}  \nАртефакты собраны: {generated} UTC",
    ),
    "missing_artifacts": (
        "No artifacts found in `{path}`. Run `python -m churn.generate_data` and "
        "`python -m churn.pipeline` first.",
        "Артефакты не найдены в `{path}`. Сначала запустите `python -m churn.generate_data` и "
        "`python -m churn.pipeline`.",
    ),
    "section_overview": ("Overview", "Обзор"),
    "section_quality": ("Model quality and best model", "Качество моделей и лучшая модель"),
    "section_pred_obs": ("Test: predicted vs observed", "Тест: прогноз против факта"),
    "section_hazard": ("Hazard: predicted vs observed", "Hazard: прогноз против факта"),
    "section_over_time": ("Brier score and AUC over time", "Brier score и AUC во времени"),
    "section_calibration": ("Calibration", "Калибровка"),
    "section_bootstrap": ("Bootstrap of metrics", "Бутстрап метрик"),
    "section_importance": ("Feature importance", "Важность признаков"),
    "section_dependence": ("Dependence plots", "Графики зависимости"),
    "section_rul": ("Remaining useful life", "Остаточный срок жизни (RUL)"),
    "section_backtest": ("Backtest", "Бэктест"),
    "section_production": ("Evaluation vs production", "Evaluation против production"),
    "section_risk": ("Riskiest players and scoring", "Самые рисковые игроки и скоринг"),
    # --- Overview -------------------------------------------------------------
    "kpi_best_model": ("Best model", "Лучшая модель"),
    "kpi_test_ibs": ("Test IBS ({variant})", "IBS на тесте ({variant})"),
    "kpi_test_auc": ("Test AUC at horizon", "AUC на тесте на горизонте"),
    "kpi_churn_rate": ("Observed churn rate, test", "Фактический отток, тест"),
    "kpi_high_risk": ("High-risk players", "Игроки высокого риска"),
    "setup_text": (
        "**Setup.** A player churns on the day that completes the first run of {inactivity} "
        "consecutive days without activity. Each row is a player who is still at risk on a "
        "monthly snapshot date; features use only the history before that date and the target "
        "is the number of days until churn, censored after {follow_up} days. Two survival "
        "models, XGBoost Cox and XGBoost AFT, predict the whole curve `S(t | x)` for days "
        "1..{grid_end}; the business score is `P(churn within {horizon} days) = 1 - S({horizon})`. "
        "A player last active `r` days before the snapshot cannot churn during the first "
        "`{inactivity} - r` days, so the models count time from the end of that period and "
        "predict `S = 1` before it. The data are synthetic.",
        "**Постановка.** Игрок считается ушедшим в день, когда завершается первая серия из "
        "{inactivity} дней подряд без активности. Каждая строка — игрок, который ещё находится "
        "под риском на дату ежемесячного среза; признаки используют только историю до этой "
        "даты, а таргет — число дней до оттока с цензурированием после {follow_up} дней. Две "
        "survival-модели, XGBoost Cox и XGBoost AFT, прогнозируют всю кривую `S(t | x)` для дней "
        "1..{grid_end}; бизнес-скор — `P(отток за {horizon} дней) = 1 - S({horizon})`. "
        "Игрок, последний раз активный за `r` дней до среза, не может уйти в первые "
        "`{inactivity} - r` дней, поэтому модели отсчитывают время от конца этого периода, "
        "а до него дают `S = 1`. Данные синтетические.",
    ),
    "datasets": ("Datasets", "Выборки"),
    "tuning": ("Hyperparameter tuning", "Тюнинг гиперпараметров"),
    "tuning_note": (
        "Optuna (TPE), {n_trials} trials per model, objective: integrated Brier score averaged "
        "over the rolling-origin folds {folds}. On each fold the model is fitted on the earlier "
        "snapshots with the labels known on the fold date.",
        "Optuna (TPE), {n_trials} испытаний на модель, целевая метрика — integrated Brier score, "
        "усреднённый по фолдам со скользящим началом {folds}. На каждом фолде модель обучается "
        "на более ранних срезах с метками, известными на дату фолда.",
    ),
    "hyperparameter": ("Hyperparameter", "Гиперпараметр"),
    "trial": ("Trial", "Испытание"),
    "trial_value": ("trial", "испытание"),
    "best_so_far": ("best so far", "лучшее на текущий момент"),
    # --- Quality --------------------------------------------------------------
    "best_model_text": (
        "Selected: **{model}, {variant}**. The criterion is the integrated Brier score averaged "
        "over the selection folds ({folds}), each forecast the way production works: models "
        "refitted on all earlier snapshots, calibrator taken from the latest earlier model "
        "whose snapshot has full follow-up. A family is calibrated only if the 95% interval of "
        "the calibration effect lies below zero; the family with the lower criterion wins. "
        "The test snapshot is used for reporting only.",
        "Выбрана: **{model}, {variant}**. Критерий — integrated Brier score, усреднённый по "
        "фолдам выбора ({folds}); каждый фолд спрогнозирован так же, как работает production: "
        "модели переобучены на всех более ранних срезах, калибратор взят от последней более "
        "ранней модели, чей срез прослежен полностью. Семейство калибруется, только если 95% "
        "интервал эффекта калибровки целиком ниже нуля; побеждает семейство с меньшим "
        "критерием. Тестовый срез используется только для отчёта.",
    ),
    "selection_title": ("Model selection", "Выбор модели"),
    "selection_note": (
        "Intervals come from {n_boot} bootstrap resamples of players, not rows: a player is "
        "at risk on several folds. The last row tells whether the folds can tell the selected "
        "candidate from the runner-up.",
        "Интервалы получены по {n_boot} бутстрап-выборкам игроков, а не строк: один игрок "
        "находится под риском на нескольких фолдах. Последняя строка показывает, различают ли "
        "фолды выбранного кандидата и следующего за ним.",
    ),
    "quality_title": ("Quality on validation and test", "Качество на валидации и тесте"),
    "quality_note": (
        "IBS is integrated over the daily grid; Brier score and AUC are taken at day {horizon}; "
        "the C-index ranks players by the predicted churn probability at day {horizon}. "
        "Two references use no model. Kaplan-Meier is the training survival curve applied to "
        "everybody. Kaplan-Meier by recency is one training curve per number of days since the "
        "last activity: what the churn definition alone gives, and the bar a model has to clear.",
        "IBS интегрируется по дневной сетке; Brier score и AUC считаются на {horizon}-й день; "
        "C-index ранжирует игроков по прогнозной вероятности оттока на {horizon}-й день. "
        "Два ориентира обходятся без модели. Каплан–Мейер — кривая обучающей выборки для всех "
        "игроков. Каплан–Мейер по давности — своя кривая обучающей выборки для каждого числа "
        "дней с последней активности: то, что даёт одно только определение оттока, и планка, "
        "которую модель должна превзойти.",
    ),
    "strata_title": ("Test quality by recency", "Качество на тесте по давности активности"),
    "strata_note": (
        "Players inactive for a long time churn almost surely and dominate the pooled metrics, "
        "so the same metrics are shown within groups of days since the last activity. Recently "
        "active players are the ones that can still be retained. Within a group the integrated "
        "Brier score covers only the days that the group's follow-up supports, so compare "
        "models within a group, not groups with each other.",
        "Давно неактивные игроки уходят почти наверняка и определяют общие метрики, поэтому "
        "те же метрики показаны внутри групп по числу дней с последней активности. Недавно "
        "активные игроки — те, кого ещё можно удержать. Внутри группы integrated Brier score "
        "покрывает только дни, обеспеченные наблюдением этой группы, поэтому сравнивайте "
        "модели внутри группы, а не группы между собой.",
    ),
    "metric": ("Metric", "Метрика"),
    "model_variant": ("Model", "Модель"),
    # --- Predicted vs observed ------------------------------------------------
    "survival_title": ("Survival curves", "Кривые выживаемости"),
    "survival_note": (
        "Mean predicted survival curve of the test players against the Kaplan-Meier estimate "
        "of what actually happened.",
        "Средняя прогнозная кривая выживаемости игроков теста против оценки Каплана–Мейера "
        "по фактическим исходам.",
    ),
    "observed_km": ("Observed (Kaplan-Meier)", "Факт (Каплан–Мейер)"),
    "day": ("Days since snapshot", "Дней с даты среза"),
    "survival_probability": ("Share of players not churned", "Доля неушедших игроков"),
    "deciles_title": ("Churn rate by risk decile", "Доля оттока по децилям риска"),
    "horizon": ("Horizon, days", "Горизонт, дней"),
    "predicted": ("Predicted", "Прогноз"),
    "observed": ("Observed", "Факт"),
    "risk_decile": (
        "Decile of predicted churn probability (10 = highest risk)",
        "Дециль прогнозной вероятности оттока (10 — максимальный риск)",
    ),
    "churn_rate_by": ("Churned within {horizon} days", "Ушли за {horizon} дней"),
    # --- Hazard ---------------------------------------------------------------
    "hazard_note": (
        "Daily churn hazard: the share of players still at risk who churn on a given day. "
        "Observed = events / players at risk; predicted = hazard of the group's mean survival "
        "curve. Risk groups are terciles of the predicted {horizon}-day churn probability. "
        "Players outside the high-risk group were active recently, so they cannot complete "
        "{inactivity} inactive days in the first weeks.",
        "Дневной hazard оттока: доля игроков под риском, уходящих в конкретный день. "
        "Факт = события / игроки под риском; прогноз = hazard средней кривой выживаемости "
        "группы. Группы риска — терцили прогнозной вероятности оттока за {horizon} дней. "
        "Игроки вне группы высокого риска были активны недавно и не могут набрать "
        "{inactivity} дней неактивности в первые недели.",
    ),
    "risk_groups": ("Risk groups", "Группы риска"),
    "smoothing_days": ("Smoothing, days", "Сглаживание, дней"),
    "daily_hazard": ("Daily hazard", "Дневной hazard"),
    # --- Over time ------------------------------------------------------------
    "over_time_note": (
        "Time-dependent metrics on the test snapshot; the dashed line marks the business "
        "horizon. Lower Brier score and higher AUC are better.",
        "Зависящие от времени метрики на тестовом срезе; пунктир отмечает бизнес-горизонт. "
        "Чем ниже Brier score и выше AUC, тем лучше.",
    ),
    "brier_title": ("Brier score over time", "Brier score во времени"),
    "auc_title": ("Cumulative/dynamic AUC over time", "Кумулятивный/динамический AUC во времени"),
    # --- Calibration ----------------------------------------------------------
    "calibration_note": (
        "Each model is recalibrated with one isotonic regression per day of the grid "
        "(1..{grid_end}), fitted on the validation snapshot and applied to the test snapshot. "
        "Points are deciles of the predicted probability; the dashed diagonal is perfect "
        "calibration. Whether the scores use the calibrated curves is decided on the selection "
        "folds, not here.",
        "Каждая модель перекалибрована изотонической регрессией для каждого дня сетки "
        "(1..{grid_end}): обучение на валидационном срезе, применение на тестовом. "
        "Точки — децили прогнозной вероятности; пунктирная диагональ — идеальная калибровка. "
        "Использовать ли калиброванные кривые в скоринге, решается на фолдах выбора, а не здесь.",
    ),
    "predicted_churn_rate": ("Predicted churn rate", "Прогнозная доля оттока"),
    "observed_churn_rate": ("Observed churn rate", "Фактическая доля оттока"),
    "calibration_effect": ("Effect of calibration on test metrics", "Эффект калибровки на тесте"),
    "yes": ("yes", "да"),
    "no": ("no", "нет"),
    # --- Bootstrap ------------------------------------------------------------
    "bootstrap_note": (
        "{n_boot} bootstrap resamples of the test players; points are the full-sample "
        "estimates and bars are 95% percentile intervals.",
        "{n_boot} бутстрап-выборок игроков теста; точки — оценки на полной выборке, "
        "отрезки — 95% перцентильные интервалы.",
    ),
    "paired_differences": ("Paired differences", "Парные разности"),
    "paired_note": (
        "Both sides of a difference are computed on the same resample. A difference is "
        "flagged as significant when its 95% interval excludes zero.",
        "Обе части разности считаются на одной и той же бутстрап-выборке. Разность помечена "
        "значимой, если её 95% интервал не содержит ноль.",
    ),
    # --- Importance and dependence ---------------------------------------------
    "importance_note": (
        "Shares within each model, so that Cox (log hazard) and AFT (log time) are comparable. "
        "SHAP values of the evaluation models are computed on the test snapshot, those of the "
        "production model on the scoring snapshot. Importance describes the time left once "
        "churn is possible; the churn-free period that recency fixes by definition is not "
        "learned and has no importance.",
        "Доли внутри каждой модели, чтобы Cox (лог-hazard) и AFT (лог-время) были сопоставимы. "
        "SHAP-значения evaluation-моделей посчитаны на тестовом срезе, production-модели — "
        "на срезе скоринга. Важность описывает время после того, как отток стал возможен; "
        "период без оттока, который давность задаёт по определению, не обучается и важности "
        "не имеет.",
    ),
    "importance_type": ("Importance type", "Тип важности"),
    "importance_share": ("Share of total importance", "Доля суммарной важности"),
    "raw_values": ("Raw values", "Исходные значения"),
    "dependence_note": (
        "SHAP dependence: how a feature's value moves the risk score (positive = towards "
        "churn), coloured by a second feature to reveal interactions. Any of the {n_pairs} "
        "ordered feature pairs can be selected.",
        "SHAP-зависимость: как значение признака сдвигает риск-скор (плюс — в сторону оттока); "
        "цвет второго признака показывает взаимодействия. Можно выбрать любую из {n_pairs} "
        "упорядоченных пар признаков.",
    ),
    "model": ("Model", "Модель"),
    "feature": ("Feature", "Признак"),
    "color_feature": ("Colour by", "Цвет по признаку"),
    "all_interactions": (
        "Show the selected feature against every other feature",
        "Показать выбранный признак со всеми остальными",
    ),
    "all_interactions_note": (
        "Each panel: {feature} on the x-axis, its SHAP value on the y-axis, colour = the "
        "feature in the panel title (dark = low, bright = high).",
        "На каждой панели: {feature} по оси X, его SHAP-значение по оси Y, цвет — признак "
        "из заголовка панели (тёмный — низкие значения, яркий — высокие).",
    ),
    "shap_axis": ("SHAP value of {feature}", "SHAP-значение: {feature}"),
    # --- RUL ------------------------------------------------------------------
    "rul_note": (
        "Remaining useful life = expected days until churn, restricted to {days} days, from "
        "the {model} curves ({variant}) on the test snapshot. Kaplan-Meier gives one number for "
        "everybody; the model gives one per player.",
        "Остаточный срок жизни — ожидаемое число дней до оттока, ограниченное {days} днями, "
        "по кривым модели {model} ({variant}) на тестовом срезе. Каплан–Мейер даёт одно "
        "число на всех; модель — своё для каждого игрока.",
    ),
    "rul_mean_predicted": ("Mean predicted RUL, days", "Средний прогнозный RUL, дней"),
    "rul_mean_observed": ("Mean observed RUL, days", "Средний фактический RUL, дней"),
    "rul_mae_model": ("MAE, model", "MAE, модель"),
    "rul_mae_km": ("MAE, Kaplan-Meier", "MAE, Каплан–Мейер"),
    "vs_km": ("vs Kaplan-Meier", "к Каплану–Мейеру"),
    "rul_model": ("Model RUL", "RUL по модели"),
    "rul_observed_km": (
        "Observed RUL (Kaplan-Meier within decile)",
        "Фактический RUL (Каплан–Мейер внутри дециля)",
    ),
    "rul_km_baseline": (
        "Kaplan-Meier RUL without a model",
        "RUL по Каплану–Мейеру без модели",
    ),
    "rul_decile": (
        "Decile of predicted RUL (1 = shortest)",
        "Дециль прогнозного RUL (1 — самый короткий)",
    ),
    "rul_axis": ("Days until churn (max {days})", "Дней до оттока (максимум {days})"),
    # --- Backtest -------------------------------------------------------------
    "backtest_note": (
        "Rolling origin: for every date the models are refitted on all earlier snapshots with "
        "the labels known on that date, then predict the {horizon}-day churn of the players at "
        "risk on that date. The calibrated variant uses the calibrator of the model fitted "
        "{lag} earlier, the latest one whose snapshot has full follow-up: exactly what the "
        "scoring of the current players does. The role column says what the main run uses a "
        "date for. Hyperparameters are tuned on the tuning folds and the family is chosen on "
        "the selection folds, so only the test date and the dates after it are untouched by "
        "every decision.",
        "Скользящее начало: для каждой даты модели переобучаются на всех более ранних срезах "
        "с метками, известными на эту дату, и прогнозируют отток за {horizon} дней для игроков "
        "под риском на эту дату. Калиброванный вариант использует калибратор модели, обученной "
        "на {lag} раньше, — последней, чей срез прослежен полностью: именно так скорятся текущие "
        "игроки. Столбец роли показывает, для чего дата используется в основном запуске. "
        "Гиперпараметры подобраны на фолдах тюнинга, а семейство выбрано на фолдах выбора, "
        "поэтому ни одно решение не затрагивает только тестовую дату и даты после неё.",
    ),
    "calibration_lag": ("{months} months", "{months} мес."),
    "snapshot_date": ("Snapshot date", "Дата среза"),
    # --- Production -----------------------------------------------------------
    "production_note": (
        "The production model is the selected family ({model}) refitted on every snapshot "
        "before the scoring date {scoring} with the labels known that day ({prod_rows:,} rows). "
        "Its calibrator belongs to the model fitted as of {calibration}, the latest one whose "
        "snapshot has full follow-up. It scores players whose outcomes are still unknown, so "
        "it cannot be tested itself. The comparison below replays the same procedure on the "
        "test date {test}: the rows named production are the model refitted on every snapshot "
        "before {test}, the rows named evaluation are the model frozen on {val} "
        "({eval_rows:,} rows); both use the calibrator fitted on the {val} snapshot.",
        "Production-модель — выбранное семейство ({model}), переобученное на всех срезах до "
        "даты скоринга {scoring} с метками, известными на этот день ({prod_rows:,} строк). "
        "Её калибратор принадлежит модели, обученной по состоянию на {calibration}, — последней, "
        "чей срез прослежен полностью. Она скорит игроков, исходы которых ещё неизвестны, "
        "поэтому саму её проверить нельзя. Сравнение ниже воспроизводит ту же процедуру на "
        "тестовую дату {test}: строки production — модель, переобученная на всех срезах до "
        "{test}, строки evaluation — модель, замороженная на {val} ({eval_rows:,} строк); "
        "обе используют калибратор, обученный на срезе {val}.",
    ),
    "agreement_rank": (
        "Spearman correlation of risk scores",
        "Корреляция Спирмена риск-скоров",
    ),
    "agreement_gap": (
        "Mean absolute gap in churn probability",
        "Среднее абсолютное расхождение вероятности оттока",
    ),
    "score_agreement_title": (
        "{horizon}-day churn probability ({variant})",
        "Вероятность оттока за {horizon} дней ({variant})",
    ),
    # --- Risk -----------------------------------------------------------------
    "risk_note": (
        "Players at risk on {date}, scored by the production model ({variant}). Players close "
        "to {inactivity} days of inactivity are almost certain to churn by definition; lower "
        "the recency filter to focus on players who can still be retained. A probability is "
        "exactly zero for as long as the player cannot complete {inactivity} inactive days.",
        "Игроки под риском на {date}, оценённые production-моделью ({variant}). Игроки, "
        "близкие к {inactivity} дням неактивности, почти наверняка уйдут по определению; "
        "уменьшите фильтр давности, чтобы сосредоточиться на тех, кого ещё можно удержать. "
        "Вероятность равна нулю, пока игрок не может набрать {inactivity} дней неактивности.",
    ),
    "risk_segment": ("Risk segment", "Сегмент риска"),
    "players": ("Players", "Игроки"),
    "churn_prob_axis": (
        "Probability of churn within {horizon} days",
        "Вероятность оттока за {horizon} дней",
    ),
    "top_title": ("Riskiest players", "Самые рисковые игроки"),
    "top_n": ("Number of players", "Число игроков"),
    "max_recency": (
        "Max days since last activity",
        "Максимум дней с последней активности",
    ),
    "no_players": ("No players match the filters.", "Нет игроков, подходящих под фильтры."),
    "scorecard_title": ("Player scorecard", "Карточка игрока"),
    "player": ("Player ID", "ID игрока"),
    "churn_within": ("Churn within {days} d", "Отток за {days} дн."),
    "player_curve": ("This player", "Этот игрок"),
    "average_curve": ("Average of all scored players", "Среднее по всем игрокам"),
    "drivers_title": ("Top risk drivers", "Главные факторы риска"),
    "drivers_note": (
        "SHAP contributions to the risk score; positive values push towards churn.",
        "SHAP-вклады в риск-скор; положительные значения сдвигают в сторону оттока.",
    ),
    "all_features": ("All features of the player", "Все признаки игрока"),
    # --- Data values: models, variants, splits, groups ------------------------
    "cox": ("XGBoost Cox", "XGBoost Cox"),
    "aft": ("XGBoost AFT", "XGBoost AFT"),
    "kaplan_meier": ("Kaplan-Meier", "Каплан–Мейер"),
    "km_recency": ("Kaplan-Meier by recency", "Каплан–Мейер по давности"),
    "evaluation": ("Evaluation model", "Evaluation-модель"),
    "production": ("Production model", "Production-модель"),
    "raw": ("raw", "без калибровки"),
    "calibrated": ("calibrated", "калиброванная"),
    "train": ("Train", "Train"),
    "val": ("Validation", "Validation"),
    "test": ("Test", "Test"),
    "production_train": ("Production train", "Production train"),
    "scoring": ("Scoring", "Скоринг"),
    "tune": ("Tuning fold", "Фолд тюнинга"),
    "selection": ("Selection fold", "Фолд выбора"),
    "later": ("After test", "После теста"),
    "other": ("Not a decision fold", "Вне решений"),
    "all": ("All players", "Все игроки"),
    "low": ("Low risk", "Низкий риск"),
    "medium": ("Medium risk", "Средний риск"),
    "high": ("High risk", "Высокий риск"),
    # --- Data values: metrics and table columns --------------------------------
    "ibs": ("Integrated Brier score", "Integrated Brier score"),
    "brier_horizon": ("Brier score at horizon", "Brier score на горизонте"),
    "auc_horizon": ("AUC at horizon", "AUC на горизонте"),
    "c_index": ("C-index", "C-index"),
    "brier": ("Brier score", "Brier score"),
    "auc": ("AUC", "AUC"),
    "ibs_horizon": ("IBS up to horizon", "IBS до горизонта"),
    "split": ("Split", "Выборка"),
    "variant": ("Variant", "Вариант"),
    "role": ("Role of the date", "Роль даты"),
    "stratum": ("Days since last activity", "Дней с последней активности"),
    "selected": ("Selected", "Выбрана"),
    "snapshot_dates": ("Snapshot dates", "Даты срезов"),
    "rows": ("Rows", "Строки"),
    "event_rate": ("Event rate", "Доля событий"),
    "churn_rate_horizon": ("Churn rate at horizon", "Отток на горизонте"),
    "mean": ("Bootstrap mean", "Среднее по бутстрапу"),
    "ci_low": ("95% CI, low", "95% ДИ, нижняя"),
    "ci_high": ("95% CI, high", "95% ДИ, верхняя"),
    "point": ("Estimate", "Оценка"),
    "difference": ("Difference", "Разность"),
    "significant": ("Significant", "Значимо"),
    "gain_share": ("Gain share", "Доля gain"),
    "mean_abs_shap": ("Mean |SHAP|", "Средний |SHAP|"),
    "bin": ("Decile", "Дециль"),
    "n": ("Players", "Игроки"),
    "km_baseline": ("Kaplan-Meier baseline", "Базовый Каплан–Мейер"),
    "train_rows": ("Training rows", "Строк в обучении"),
    "predicted_rate": ("Predicted churn rate", "Прогнозная доля оттока"),
    "observed_rate": ("Observed churn rate", "Фактическая доля оттока"),
    "risk_rank": ("Rank", "Ранг"),
    "player_id": ("Player ID", "ID игрока"),
    "churn_prob_7d": ("P(churn) 7 d", "P(отток) 7 дн."),
    "churn_prob_14d": ("P(churn) 14 d", "P(отток) 14 дн."),
    "churn_prob_30d": ("P(churn) 30 d", "P(отток) 30 дн."),
    "churn_prob_60d": ("P(churn) 60 d", "P(отток) 60 дн."),
    "rul_days": ("RUL, days", "RUL, дней"),
    "driver_1": ("Driver 1", "Фактор 1"),
    "driver_2": ("Driver 2", "Фактор 2"),
    "driver_3": ("Driver 3", "Фактор 3"),
    "value": ("Value", "Значение"),
    "shap": ("SHAP", "SHAP"),
    # --- Data values: features and categories ----------------------------------
    "tenure_days": ("Tenure, days", "Стаж, дней"),
    "recency_days": ("Days since last activity", "Дней с последней активности"),
    "active_days_7d": ("Active days, last 7", "Активных дней за 7"),
    "active_days_30d": ("Active days, last 30", "Активных дней за 30"),
    "activity_ratio_7d_30d": ("Activity ratio 7d / 30d", "Отношение активности 7д / 30д"),
    "sessions_30d": ("Sessions, 30 d", "Сессии за 30 дн."),
    "turnover_30d": ("Turnover, 30 d", "Оборот за 30 дн."),
    "ggr_30d": ("GGR, 30 d", "GGR за 30 дн."),
    "deposits_30d": ("Deposits, 30 d", "Депозиты за 30 дн."),
    "withdrawals_30d": ("Withdrawals, 30 d", "Выводы за 30 дн."),
    "bonus_30d": ("Bonuses, 30 d", "Бонусы за 30 дн."),
    "avg_stake_30d": ("Average stake, 30 d", "Средняя ставка за 30 дн."),
    "age": ("Age", "Возраст"),
    "acquisition_channel": ("Acquisition channel", "Канал привлечения"),
    "favorite_game": ("Favourite game", "Любимая игра"),
    "organic": ("Organic", "Органика"),
    "affiliate": ("Affiliate", "Партнёры"),
    "paid_ads": ("Paid ads", "Платная реклама"),
    "referral": ("Referral", "Рефералы"),
    "slots": ("Slots", "Слоты"),
    "live_casino": ("Live casino", "Live-казино"),
    "table_games": ("Table games", "Настольные игры"),
    "crash_games": ("Crash games", "Crash-игры"),
}


class Translator:
    """Look up dashboard strings in one language.

    Parameters
    ----------
    language : str
        ``"en"`` or ``"ru"``.
    """

    def __init__(self, language: str) -> None:
        self._index = list(LANGUAGES.values()).index(language)

    def __call__(self, key: str, **values: object) -> str:
        """Translate a UI message id, filling ``{placeholders}`` from ``values``.

        Raises
        ------
        KeyError
            If ``key`` is unknown: a missing UI string is a bug, not a fallback case.
        """
        return TEXT[key][self._index].format(**values)

    def label(self, value: str) -> str:
        """Readable label of a data value; values without a label are shown as is."""
        entry = TEXT.get(value)
        return entry[self._index] if entry else value
