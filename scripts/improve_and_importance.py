"""Run focused model improvements and variable-importance reports."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stat390-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/stat390-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from evaluate import evaluate_classification
from model import RANDOM_STATE, build_preprocessor
from run_full_experiments import (
    BURNOUT_TARGET,
    TREATMENT_TARGET,
    build_burnout_frame,
    build_survey_frame,
    burnout_features,
    evaluate_regression,
    survey_features,
)


RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("figures")
IMPROVEMENT_LOG_PATH = RESULTS_DIR / "focused_improvement_log.csv"
SUMMARY_PATH = RESULTS_DIR / "focused_improvement_summary.json"
IMPORTANCE_PATH = RESULTS_DIR / "variable_importance.csv"
IMPORTANCE_REPORT_PATH = REPORTS_DIR / "variable_importance.md"
IMPROVEMENT_REPORT_PATH = REPORTS_DIR / "focused_model_improvement.md"


def classification_candidates() -> list[tuple[str, object]]:
    return [
        (
            "Logistic_C0.03_balanced",
            LogisticRegression(C=0.03, max_iter=5000, class_weight="balanced"),
        ),
        (
            "Logistic_C0.05_balanced",
            LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"),
        ),
        (
            "Logistic_C0.2_balanced",
            LogisticRegression(C=0.2, max_iter=5000, class_weight="balanced"),
        ),
        (
            "HistGB_lr003_leaf15_l2",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "HistGB_lr002_leaf31_l2",
            HistGradientBoostingClassifier(
                learning_rate=0.02,
                max_iter=450,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "NeuralNet_MLP_48_24",
            MLPClassifier(
                hidden_layer_sizes=(48, 24),
                alpha=0.005,
                max_iter=350,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def regression_candidates() -> list[tuple[str, object]]:
    return [
        ("Ridge_alpha0.1", Ridge(alpha=0.1)),
        ("Ridge_alpha3", Ridge(alpha=3.0)),
        (
            "HistGBReg_lr003_leaf31_l2",
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=350,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "HistGBReg_lr002_leaf63_l2",
            HistGradientBoostingRegressor(
                learning_rate=0.02,
                max_iter=500,
                max_leaf_nodes=63,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "HistGBReg_lr005_leaf15_l2",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "NeuralNetReg_MLP_48_24",
            MLPRegressor(
                hidden_layer_sizes=(48, 24),
                alpha=0.005,
                max_iter=350,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def run_candidates(run_id: str, task_type: str, X: pd.DataFrame, y: pd.Series) -> tuple[list[dict], dict]:
    if task_type == "classification":
        split = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        candidates = classification_candidates()
    else:
        split = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        candidates = regression_candidates()

    X_train, X_val, y_train, y_val = split
    rows = []
    fitted = {}

    for model_name, estimator in candidates:
        start = time.time()
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
                "run_id": run_id,
                "task_type": task_type,
                "model": model_name,
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
            }
        else:
            metrics = evaluate_regression(y_val, predictions)
            primary_metric = metrics["r2"]
            row = {
                "run_id": run_id,
                "task_type": task_type,
                "model": model_name,
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
            }

        rows.append(row)
        fitted[model_name] = {
            "pipeline": pipeline,
            "X_val": X_val,
            "y_val": y_val,
        }
        print(f"{run_id} | {model_name} | primary={primary_metric:.4f} | runtime={runtime:.2f}s")

    return rows, fitted


def score_for_importance(task_type: str):
    if task_type == "classification":
        return "f1_weighted"
    return make_scorer(r2_score)


def compute_importance(
    run_id: str,
    task_type: str,
    model_name: str,
    fitted_payload: dict,
) -> pd.DataFrame:
    X_val = fitted_payload["X_val"]
    y_val = fitted_payload["y_val"]
    if len(X_val) > 5000:
        X_eval = X_val.sample(n=5000, random_state=RANDOM_STATE)
        y_eval = y_val.loc[X_eval.index]
    else:
        X_eval = X_val
        y_eval = y_val

    result = permutation_importance(
        fitted_payload["pipeline"],
        X_eval,
        y_eval,
        scoring=score_for_importance(task_type),
        n_repeats=5 if task_type == "classification" else 3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "run_id": run_id,
            "task_type": task_type,
            "model": model_name,
            "feature": X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False)


def write_importance_plot(importance: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, group in importance.groupby("run_id", sort=False):
        top = group.sort_values("importance_mean", ascending=False).head(12).sort_values("importance_mean")
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["importance_mean"], color="#2563eb")
        plt.xlabel("Permutation importance")
        plt.title(f"Top variable importance: {run_id}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{run_id}_variable_importance.png", dpi=180, bbox_inches="tight")
        plt.close()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(rows: list[dict], importance: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    best = (
        results.sort_values(["run_id", "primary_metric"], ascending=[True, False])
        .groupby("run_id", as_index=False)
        .head(1)
    )
    IMPROVEMENT_REPORT_PATH.write_text(
        "# Focused Model Improvement\n\n"
        "This pass tested regularized logistic, histogram-gradient-boosting, ridge, and MLP variants "
        "using the same validation split policy as the controlled experiment.\n\n"
        "## Best Focused Candidates\n\n"
        + markdown_table(
            best,
            ["run_id", "task_type", "model", "primary_metric", "accuracy", "f1_weighted", "rmse", "mae", "r2"],
        )
        + "\n",
    )

    parts = ["# Variable Importance\n"]
    for run_id, group in importance.groupby("run_id", sort=False):
        parts.extend(
            [
                f"## {run_id}",
                "",
                markdown_table(
                    group.sort_values("importance_mean", ascending=False).head(12),
                    ["feature", "importance_mean", "importance_std", "model"],
                ),
                "",
            ]
        )
    IMPORTANCE_REPORT_PATH.write_text("\n".join(parts) + "\n")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    survey = build_survey_frame()
    burnout = build_burnout_frame()
    run_specs = [
        ("treatment_non_weather", "classification", survey_features(survey, include_weather=False), survey[TREATMENT_TARGET].astype(int)),
        ("treatment_weather_augmented", "classification", survey_features(survey, include_weather=True), survey[TREATMENT_TARGET].astype(int)),
        ("burnout_index_non_weather", "regression", burnout_features(burnout, include_weather=False), burnout[BURNOUT_TARGET].astype(float)),
        ("burnout_index_weather_augmented", "regression", burnout_features(burnout, include_weather=True), burnout[BURNOUT_TARGET].astype(float)),
    ]

    all_rows = []
    all_importance = []
    summary = {"runs": []}
    for run_id, task_type, X, y in run_specs:
        print(f"=== {run_id}: focused improvement ===")
        rows, fitted = run_candidates(run_id, task_type, X, y)
        all_rows.extend(rows)
        best_row = max(rows, key=lambda row: row["primary_metric"])
        all_importance.append(
            compute_importance(run_id, task_type, best_row["model"], fitted[best_row["model"]])
        )
        summary["runs"].append(best_row)

    results = pd.DataFrame(all_rows)
    importance = pd.concat(all_importance, ignore_index=True)
    results.to_csv(IMPROVEMENT_LOG_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    write_importance_plot(importance)
    write_reports(all_rows, importance)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {IMPROVEMENT_LOG_PATH}")
    print(f"Wrote {IMPORTANCE_PATH}")
    print(f"Wrote {IMPROVEMENT_REPORT_PATH}")
    print(f"Wrote {IMPORTANCE_REPORT_PATH}")


if __name__ == "__main__":
    main()
