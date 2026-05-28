"""Run full treatment and burnout-index experiment suites.

Outputs are append-only historical logs plus regenerated report files:

- reports/experiment_result_matrix.csv
- reports/experiment_result_matrix.md
- reports/best_result_vs_baseline.md
- reports/complete_experiment_log_bundle.md
- reports/controlled_experiment_set.md
- reports/error_taxonomy.md
- reports/failure_analysis_memo.md
- reports/keep_discard_crash_summary.md
- reports/what_actually_worked_memo.md
- reports/metric_trajectory_plot.svg
- reports/metric_over_time_plot.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from burnout_index import (
    BURNOUT_INDICATORS,
    DEFAULT_OUTPUT_PATH as BURNOUT_INDEX_PATH,
    construct_burnout_index,
)
from evaluate import evaluate_classification
from model import RANDOM_STATE, build_preprocessor
from weather_merge import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SURVEY_OUTPUT_PATH,
    DEFAULT_WEATHER_PATH,
    WEATHER_FEATURE_CANDIDATES,
    run_weather_merge,
)


REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")
SLEEP_DATA_PATH = Path("data/raw/sleep_health_dataset.csv")
SURVEY_WEATHER_PATH = DEFAULT_SURVEY_OUTPUT_PATH
HISTORICAL_LOG_PATH = RESULTS_DIR / "historical_experiment_log.csv"
RUN_SUMMARY_PATH = RESULTS_DIR / "latest_full_experiment_summary.json"

TREATMENT_TARGET = "sought_treatment"
BURNOUT_TARGET = "burnout_index"

SURVEY_EXCLUDE_ALWAYS = {
    TREATMENT_TARGET,
    "treatment",
    "timestamp",
    "comments",
    "country",
}

SURVEY_WEATHER_COLUMNS = set(WEATHER_FEATURE_CANDIDATES) | {
    "daylight_hours",
    "climate_region",
    "temperature_band",
    "city",
    "latitude",
    "longitude",
    "start_date",
    "end_date",
    "retrieved_at_utc",
}

BURNOUT_WEATHER_COLUMNS = {
    "room_temperature_celsius",
    "season",
}

BURNOUT_EXCLUDE_ALWAYS = {
    BURNOUT_TARGET,
    "burnout_pc1",
    "burnout_pc2",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def classification_models() -> list[tuple[str, object]]:
    return [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("Logistic_C0.1", LogisticRegression(C=0.1, max_iter=2500, class_weight="balanced")),
        ("Logistic_C1", LogisticRegression(C=1.0, max_iter=2500, class_weight="balanced")),
        ("Logistic_C10", LogisticRegression(C=10.0, max_iter=2500, class_weight="balanced")),
        ("Ridge_alpha1", RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ("Ridge_alpha10", RidgeClassifier(alpha=10.0, class_weight="balanced")),
        ("SGDLogistic_alpha0001", SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=2500, random_state=RANDOM_STATE, class_weight="balanced")),
        ("SGDLogistic_alpha001", SGDClassifier(loss="log_loss", alpha=0.001, max_iter=2500, random_state=RANDOM_STATE, class_weight="balanced")),
        ("LinearSVC_C0.5", LinearSVC(C=0.5, random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000)),
        ("LinearSVC_C1", LinearSVC(C=1.0, random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000)),
        ("KNN_k5", KNeighborsClassifier(n_neighbors=5)),
        ("KNN_k15", KNeighborsClassifier(n_neighbors=15)),
        ("DecisionTree_depth4", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE, class_weight="balanced")),
        ("DecisionTree_depth8", DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE, class_weight="balanced")),
        ("RandomForest_depth8", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("RandomForest_full", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("ExtraTrees_depth8", ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("ExtraTrees_full", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ("GradientBoosting_lr005", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=RANDOM_STATE)),
        ("NeuralNet_MLP_32", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.001, max_iter=250, early_stopping=True, random_state=RANDOM_STATE)),
        ("NeuralNet_MLP_64_32", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=250, early_stopping=True, random_state=RANDOM_STATE)),
    ]


def regression_models() -> list[tuple[str, object]]:
    return [
        ("DummyMean", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("Ridge_alpha1", Ridge(alpha=1.0)),
        ("Ridge_alpha10", Ridge(alpha=10.0)),
        ("Lasso_alpha0001", Lasso(alpha=0.0001, max_iter=5000, random_state=RANDOM_STATE)),
        ("Lasso_alpha001", Lasso(alpha=0.001, max_iter=5000, random_state=RANDOM_STATE)),
        ("ElasticNet_alpha001", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)),
        ("ElasticNet_alpha01", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)),
        ("SGDReg_alpha0001", SGDRegressor(alpha=0.0001, max_iter=2500, random_state=RANDOM_STATE)),
        ("SGDReg_alpha001", SGDRegressor(alpha=0.001, max_iter=2500, random_state=RANDOM_STATE)),
        ("LinearSVR_C0.5", LinearSVR(C=0.5, max_iter=5000, random_state=RANDOM_STATE)),
        ("LinearSVR_C1", LinearSVR(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
        ("KNNReg_k5", KNeighborsRegressor(n_neighbors=5)),
        ("KNNReg_k15", KNeighborsRegressor(n_neighbors=15)),
        ("DecisionTreeReg_depth6", DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE)),
        ("RandomForestReg_depth8", RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ("RandomForestReg_full", RandomForestRegressor(n_estimators=160, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
        ("ExtraTreesReg_depth8", ExtraTreesRegressor(n_estimators=120, max_depth=8, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1)),
        ("GradientBoostingReg", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE)),
        ("HistGradientBoostingReg", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=RANDOM_STATE)),
        ("NeuralNetReg_MLP_32", MLPRegressor(hidden_layer_sizes=(32,), alpha=0.001, max_iter=250, early_stopping=True, random_state=RANDOM_STATE)),
        ("NeuralNetReg_MLP_64_32", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=250, early_stopping=True, random_state=RANDOM_STATE)),
    ]


def drop_empty_columns(X: pd.DataFrame) -> pd.DataFrame:
    return X.dropna(axis=1, how="all")


def build_survey_frame() -> pd.DataFrame:
    if not SURVEY_WEATHER_PATH.exists():
        run_weather_merge(
            dataset="survey",
            input_path=Path("data/raw/survey.csv"),
            weather_path=DEFAULT_WEATHER_PATH,
            output_path=SURVEY_WEATHER_PATH,
            metadata_path=DEFAULT_METADATA_PATH,
        )
    return pd.read_csv(SURVEY_WEATHER_PATH)


def build_burnout_frame() -> pd.DataFrame:
    if not BURNOUT_INDEX_PATH.exists():
        construct_burnout_index()

    sleep = pd.read_csv(SLEEP_DATA_PATH)
    index = pd.read_csv(BURNOUT_INDEX_PATH)
    return sleep.merge(index, on="person_id", how="inner", validate="one_to_one")


def survey_features(df: pd.DataFrame, include_weather: bool) -> pd.DataFrame:
    excluded = set(SURVEY_EXCLUDE_ALWAYS)
    if not include_weather:
        excluded.update(SURVEY_WEATHER_COLUMNS)

    columns = [
        column
        for column in df.columns
        if column not in excluded
        and column.split(".")[0] not in excluded
        and not column.startswith("weather_")
    ]
    return drop_empty_columns(df[columns].copy())


def burnout_features(df: pd.DataFrame, include_weather: bool) -> pd.DataFrame:
    pca_source_columns = {
        indicator["aliases"][0] for indicator in BURNOUT_INDICATORS.values()
    }
    excluded = set(BURNOUT_EXCLUDE_ALWAYS) | pca_source_columns
    if not include_weather:
        excluded.update(BURNOUT_WEATHER_COLUMNS)

    columns = [column for column in df.columns if column not in excluded]
    return drop_empty_columns(df[columns].copy())


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_models(
    run_id: str,
    target_name: str,
    task_type: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[list[dict], list[dict]]:
    if task_type == "classification":
        split = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        model_specs = classification_models()
    else:
        split = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        model_specs = regression_models()

    X_train, X_val, y_train, y_val = split
    completed = []
    failures = []
    model_count = len(model_specs)

    for model_index, (model_name, estimator) in enumerate(model_specs, start=1):
        start = time.time()
        row_base = {
            "run_timestamp": utc_timestamp(),
            "run_id": run_id,
            "target": target_name,
            "task_type": task_type,
            "model_index": model_index,
            "model": model_name,
            "n_features": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
            "random_state": RANDOM_STATE,
        }
        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_train)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)
            runtime = time.time() - start

            if task_type == "classification":
                metrics = evaluate_classification(y_val, predictions)
                primary_metric = metrics["f1_weighted"]
                row = {
                    **row_base,
                    "status": "complete",
                    "runtime_seconds": runtime,
                    "primary_metric": primary_metric,
                    "accuracy": metrics["accuracy"],
                    "f1_weighted": metrics["f1_weighted"],
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"],
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "error": None,
                }
            else:
                metrics = evaluate_regression(y_val, predictions)
                primary_metric = metrics["r2"]
                row = {
                    **row_base,
                    "status": "complete",
                    "runtime_seconds": runtime,
                    "primary_metric": primary_metric,
                    "accuracy": None,
                    "f1_weighted": None,
                    "precision_weighted": None,
                    "recall_weighted": None,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "error": None,
                }

            completed.append(row)
            print(
                f"{run_id} | {model_index:02d}/{model_count} | {model_name} | "
                f"primary={primary_metric:.4f} | runtime={runtime:.2f}s"
            )
        except Exception as exc:
            runtime = time.time() - start
            failures.append(
                {
                    **row_base,
                    "status": "failed",
                    "runtime_seconds": runtime,
                    "primary_metric": None,
                    "accuracy": None,
                    "f1_weighted": None,
                    "precision_weighted": None,
                    "recall_weighted": None,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "error": repr(exc),
                }
            )
            print(f"{run_id} | {model_index:02d}/{model_count} | {model_name} | failed: {exc}")

    return completed, failures


def append_historical_log(rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if HISTORICAL_LOG_PATH.exists():
        old = pd.read_csv(HISTORICAL_LOG_PATH)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(HISTORICAL_LOG_PATH, index=False)


def write_latest_results(rows: list[dict], failures: list[dict]) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows + failures)
    results.to_csv(REPORTS_DIR / "experiment_result_matrix.csv", index=False)
    results.to_csv(RESULTS_DIR / "latest_experiment_result_matrix.csv", index=False)
    return results


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows.\n"
    view = df[columns].copy()
    headers = [str(column) for column in view.columns]
    rows = []
    for _, row in view.iterrows():
        rows.append(
            [
                format_float(value) if isinstance(value, float) else str(value)
                for value in row.tolist()
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_float(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def best_rows(results: pd.DataFrame) -> pd.DataFrame:
    complete = results[results["status"].eq("complete")].copy()
    if complete.empty:
        return complete
    return (
        complete.sort_values(["run_id", "primary_metric"], ascending=[True, False])
        .groupby("run_id", as_index=False)
        .head(1)
    )


def baseline_rows(results: pd.DataFrame) -> pd.DataFrame:
    complete = results[results["status"].eq("complete")].copy()
    return complete[complete["model_index"].eq(1)].copy()


def write_experiment_matrix_md(results: pd.DataFrame) -> None:
    complete = results[results["status"].eq("complete")].copy()
    complete = complete.sort_values(["run_id", "primary_metric"], ascending=[True, False])
    columns = [
        "run_id",
        "model_index",
        "model",
        "n_features",
        "primary_metric",
        "accuracy",
        "f1_weighted",
        "rmse",
        "mae",
        "r2",
        "runtime_seconds",
    ]
    body = [
        "# Experiment-Result Matrix",
        "",
        markdown_table(complete, columns),
        "",
        "## Summary",
        "",
        "Each target is run twice: first with non-weather predictors only, then with weather-augmented predictors.",
        "Classification uses weighted F1 as the primary metric. Burnout-index regression uses R-squared as the primary metric.",
    ]
    (REPORTS_DIR / "experiment_result_matrix.md").write_text("\n".join(body) + "\n")


def write_best_result_vs_baseline(results: pd.DataFrame) -> None:
    best = best_rows(results)
    base = baseline_rows(results)
    rows = []
    for _, best_row in best.iterrows():
        base_match = base[base["run_id"].eq(best_row["run_id"])]
        if base_match.empty:
            continue
        baseline = base_match.iloc[0]
        rows.append(
            {
                "run_id": best_row["run_id"],
                "baseline_model": baseline["model"],
                "best_model": best_row["model"],
                "baseline_primary": baseline["primary_metric"],
                "best_primary": best_row["primary_metric"],
                "delta": best_row["primary_metric"] - baseline["primary_metric"],
            }
        )
    compare = pd.DataFrame(rows)
    body = ["# Best Result vs. Baseline", ""]
    body.append(markdown_table(compare, compare.columns.tolist()) if not compare.empty else "No completed comparisons.")
    body.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Keep the best model only when it improves the run's primary validation metric over the dummy baseline.",
        ]
    )
    (REPORTS_DIR / "best_result_vs_baseline.md").write_text("\n".join(body) + "\n")


def write_controlled_experiment_set(results: pd.DataFrame) -> None:
    body = [
        "# Controlled Experiment Set",
        "",
        "- `treatment_non_weather`: predict whether the respondent sought treatment using survey variables only.",
        "- `treatment_weather_augmented`: add state-level weather features.",
        "- `burnout_index_non_weather`: predict PCA burnout index without PCA source variables or weather variables.",
        "- `burnout_index_weather_augmented`: add weather variables from the sleep dataset.",
        "",
        "Each run uses the same fixed random seed and the same validation split policy.",
        "",
        markdown_table(
            results[["run_id", "target", "task_type", "model_index", "model", "status"]],
            ["run_id", "target", "task_type", "model_index", "model", "status"],
        ),
    ]
    (REPORTS_DIR / "controlled_experiment_set.md").write_text("\n".join(body) + "\n")


def write_error_and_failure_reports(results: pd.DataFrame) -> None:
    failures = results[results["status"].ne("complete")].copy()
    failure_table = markdown_table(
        failures,
        ["run_id", "model_index", "model", "runtime_seconds", "error"],
    ) if not failures.empty else "No model failures in the latest run."

    (REPORTS_DIR / "error_taxonomy.md").write_text(
        "# Error Taxonomy\n\n"
        "## Latest Run Failures\n\n"
        f"{failure_table}\n"
    )
    (REPORTS_DIR / "failure_analysis_memo.md").write_text(
        "# Failure Analysis Memo\n\n"
        f"{failure_table}\n\n"
        "Warnings from sklearn that do not stop a run are not counted as failures.\n"
    )


def write_keep_discard_summary(results: pd.DataFrame) -> None:
    complete = results[results["status"].eq("complete")].copy()
    best = best_rows(results)
    best_keys = set(zip(best["run_id"], best["model"]))
    rows = []
    for _, row in complete.iterrows():
        rows.append(
            {
                "run_id": row["run_id"],
                "model": row["model"],
                "primary_metric": row["primary_metric"],
                "decision": "Keep" if (row["run_id"], row["model"]) in best_keys else "Discard",
            }
        )
    table = pd.DataFrame(rows)
    (REPORTS_DIR / "keep_discard_crash_summary.md").write_text(
        "# Keep / Discard / Crash Summary\n\n"
        + markdown_table(table, ["run_id", "model", "primary_metric", "decision"])
        + "\n"
    )


def write_worked_memo(results: pd.DataFrame) -> None:
    best = best_rows(results)
    lines = ["# What Actually Worked Memo", ""]
    for _, row in best.iterrows():
        lines.append(
            f"- `{row['run_id']}`: `{row['model']}` was best with primary metric "
            f"{format_float(row['primary_metric'])}."
        )
    lines.extend(
        [
            "",
            "Weather improves a run only if the best weather-augmented primary metric exceeds the best non-weather primary metric for the same target.",
        ]
    )
    (REPORTS_DIR / "what_actually_worked_memo.md").write_text("\n".join(lines) + "\n")


def write_complete_bundle(results: pd.DataFrame) -> None:
    best = best_rows(results)
    body = [
        "# Complete Experiment Log Bundle",
        "",
        "## Bundle Contents",
        "",
        "- Source log: `results/historical_experiment_log.csv`",
        "- Latest matrix: `reports/experiment_result_matrix.csv`",
        "- Result matrix: `reports/experiment_result_matrix.md`",
        "- Metric plots: `reports/metric_trajectory_plot.svg`, `reports/metric_over_time_plot.svg`",
        "- Keep/discard/crash summary: `reports/keep_discard_crash_summary.md`",
        "- Best result comparison: `reports/best_result_vs_baseline.md`",
        "",
        "## Latest Best Models",
        "",
        markdown_table(
            best,
            ["run_id", "target", "task_type", "model", "primary_metric", "accuracy", "f1_weighted", "rmse", "mae", "r2"],
        ),
        "",
        "## Reproducibility Rules",
        "",
        "- Fixed random seed: `42`.",
        "- Latest runs use validation metrics only.",
        "- Historical runs are appended to `results/historical_experiment_log.csv`.",
    ]
    (REPORTS_DIR / "complete_experiment_log_bundle.md").write_text("\n".join(body) + "\n")


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{point_text}" />'


def write_metric_plot(results: pd.DataFrame, path: Path, title: str) -> None:
    complete = results[results["status"].eq("complete")].copy()
    complete = complete.sort_values(["run_id", "model_index"])
    width, height = 1100, 520
    margin = 60
    min_metric = complete["primary_metric"].min()
    max_metric = complete["primary_metric"].max()
    if pd.isna(min_metric) or pd.isna(max_metric) or min_metric == max_metric:
        min_metric, max_metric = 0.0, 1.0

    groups = list(complete["run_id"].unique())
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    max_model_index = max(int(complete["model_index"].max()), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]

    for group_index, group in enumerate(groups):
        group_df = complete[complete["run_id"].eq(group)]
        points = []
        for _, row in group_df.iterrows():
            index_position = 0 if max_model_index == 1 else (row["model_index"] - 1) / (max_model_index - 1)
            x = margin + index_position * (width - 2 * margin)
            scaled = (row["primary_metric"] - min_metric) / (max_metric - min_metric)
            y = (height - margin) - scaled * (height - 2 * margin)
            points.append((x, y))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[group_index % len(colors)]}"/>')
        parts.append(svg_polyline(points, colors[group_index % len(colors)]))
        parts.append(
            f'<text x="{width - 360}" y="{70 + group_index * 22}" font-family="Arial" font-size="13" fill="{colors[group_index % len(colors)]}">{group}</text>'
        )

    parts.append(f'<text x="{margin}" y="{height - 18}" font-family="Arial" font-size="12">Model index</text>')
    parts.append(f'<text x="12" y="{margin}" font-family="Arial" font-size="12">Primary metric</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_reports(results: pd.DataFrame) -> None:
    write_experiment_matrix_md(results)
    write_best_result_vs_baseline(results)
    write_controlled_experiment_set(results)
    write_error_and_failure_reports(results)
    write_keep_discard_summary(results)
    write_worked_memo(results)
    write_complete_bundle(results)
    write_metric_plot(results, REPORTS_DIR / "metric_trajectory_plot.svg", "Metric Trajectory by Model Index")
    write_metric_plot(results, REPORTS_DIR / "metric_over_time_plot.svg", "Latest Run Primary Metrics")


def run_all() -> pd.DataFrame:
    survey = build_survey_frame()
    burnout = build_burnout_frame()

    runs = [
        (
            "treatment_non_weather",
            TREATMENT_TARGET,
            "classification",
            survey_features(survey, include_weather=False),
            survey[TREATMENT_TARGET].astype(int),
        ),
        (
            "treatment_weather_augmented",
            TREATMENT_TARGET,
            "classification",
            survey_features(survey, include_weather=True),
            survey[TREATMENT_TARGET].astype(int),
        ),
        (
            "burnout_index_non_weather",
            BURNOUT_TARGET,
            "regression",
            burnout_features(burnout, include_weather=False),
            burnout[BURNOUT_TARGET].astype(float),
        ),
        (
            "burnout_index_weather_augmented",
            BURNOUT_TARGET,
            "regression",
            burnout_features(burnout, include_weather=True),
            burnout[BURNOUT_TARGET].astype(float),
        ),
    ]

    all_rows = []
    all_failures = []
    summary = {
        "run_timestamp": utc_timestamp(),
        "random_state": RANDOM_STATE,
        "runs": [],
    }

    for run_id, target, task_type, X, y in runs:
        print(f"=== {run_id}: {X.shape[0]} rows, {X.shape[1]} features ===")
        rows, failures = run_models(run_id, target, task_type, X, y)
        all_rows.extend(rows)
        all_failures.extend(failures)
        summary["runs"].append(
            {
                "run_id": run_id,
                "target": target,
                "task_type": task_type,
                "rows": int(X.shape[0]),
                "features": int(X.shape[1]),
                "completed": len(rows),
                "failed": len(failures),
            }
        )

    append_historical_log(all_rows + all_failures)
    results = write_latest_results(all_rows, all_failures)
    write_reports(results)

    best = best_rows(results)
    summary["best_by_run"] = best.to_dict(orient="records")
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all controlled experiment suites.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    results = run_all()
    completed = int(results["status"].eq("complete").sum())
    failed = int(results["status"].ne("complete").sum())
    print(f"Wrote latest experiment reports for {completed} completed models and {failed} failures.")


if __name__ == "__main__":
    main()
