"""Run validation experiments using frozen prepared data and editable models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from model import (
    build_boosted_tree_search_space,
    build_extra_trees_search_space,
    build_hist_gradient_boosting_search_space,
    build_logistic_regression,
    build_polynomial_logistic_search_space,
    build_random_forest_search_space,
    build_tuned_logistic_search_space,
)
from evaluate import (
    build_validation_diagnostics,
    evaluate_classification,
)
from prepare import (
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COL,
    load_prepared_data,
    prepare_data,
)


def print_metrics(title: str, metrics: dict, runtime_seconds: float) -> None:
    print(f"=== {title} VALIDATION RESULTS ===")
    print(f"Validation ACC: {metrics['accuracy']:.4f}")
    print(f"Validation F1:  {metrics['f1_weighted']:.4f}")
    print(f"Validation Precision:  {metrics['precision_weighted']:.4f}")
    print(f"Validation Recall: {metrics['recall_weighted']:.4f}")
    print(f"Runtime (seconds): {runtime_seconds:.4f}")


def run_grid_search(
    name: str,
    pipeline,
    param_grid: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    scoring: str = "f1_weighted",
):
    start_time = time.time()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=3,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    val_preds = search.predict(X_val)
    metrics = evaluate_classification(y_val, val_preds)
    runtime_seconds = time.time() - start_time

    print(f"=== {name} VALIDATION RESULTS ===")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV {scoring}: {search.best_score_:.4f}")
    print(f"Validation ACC: {metrics['accuracy']:.4f}")
    print(f"Validation F1:  {metrics['f1_weighted']:.4f}")
    print(f"Validation Precision:  {metrics['precision_weighted']:.4f}")
    print(f"Validation Recall: {metrics['recall_weighted']:.4f}")
    print(f"Runtime (seconds): {runtime_seconds:.4f}")

    return search, val_preds, metrics, runtime_seconds


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Regenerating prepared data with current feature engineering.")
    prepare_data()

    X_train, X_val, y_train, y_val = load_prepared_data()
    print("Target distribution in training data:")
    print(y_train.value_counts(normalize=True).sort_index())

    logistic_start_time = time.time()
    baseline_model = build_logistic_regression(X_train)
    baseline_model.fit(X_train, y_train)
    baseline_val_preds = baseline_model.predict(X_val)
    baseline_metrics = evaluate_classification(y_val, baseline_val_preds)
    baseline_runtime_seconds = time.time() - logistic_start_time
    print_metrics("BASELINE", baseline_metrics, baseline_runtime_seconds)
    print("Test set is locked and is not used during the search phase.")

    tuned_logistic_pipeline, tuned_logistic_param_grid = build_tuned_logistic_search_space(
        X_train
    )
    (
        tuned_logistic_search,
        tuned_logistic_val_preds,
        tuned_logistic_metrics,
        tuned_logistic_runtime_seconds,
    ) = run_grid_search(
        "TUNED LOGISTIC",
        tuned_logistic_pipeline,
        tuned_logistic_param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scoring="accuracy",
    )

    rf_pipeline, rf_param_grid = build_random_forest_search_space(X_train)
    rf_search, rf_val_preds, rf_metrics, rf_runtime_seconds = run_grid_search(
        "RANDOM FOREST",
        rf_pipeline,
        rf_param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    extra_trees_pipeline, extra_trees_param_grid = build_extra_trees_search_space(X_train)
    (
        extra_trees_search,
        extra_trees_val_preds,
        extra_trees_metrics,
        extra_trees_runtime_seconds,
    ) = run_grid_search(
        "EXTRA TREES",
        extra_trees_pipeline,
        extra_trees_param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scoring="accuracy",
    )

    poly_pipeline, poly_param_grid = build_polynomial_logistic_search_space(X_train)
    poly_search, poly_val_preds, poly_metrics, poly_runtime_seconds = run_grid_search(
        "POLYNOMIAL LOGISTIC",
        poly_pipeline,
        poly_param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    boosted_pipeline, boosted_param_grid = build_boosted_tree_search_space(X_train)
    boosted_search, boosted_val_preds, boosted_metrics, boosted_runtime_seconds = (
        run_grid_search(
            "BOOSTED TREE",
            boosted_pipeline,
            boosted_param_grid,
            X_train,
            y_train,
            X_val,
            y_val,
        )
    )

    hist_pipeline, hist_param_grid = build_hist_gradient_boosting_search_space(X_train)
    hist_search, hist_val_preds, hist_metrics, hist_runtime_seconds = run_grid_search(
        "HIST GRADIENT BOOSTING",
        hist_pipeline,
        hist_param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scoring="accuracy",
    )

    validation_diagnostics = build_validation_diagnostics(
        y_val,
        {
            "LogisticRegression": baseline_val_preds,
            "TunedLogisticRegression": tuned_logistic_val_preds,
            "RandomForestClassifier": rf_val_preds,
            "ExtraTreesClassifier": extra_trees_val_preds,
            "PolynomialLogisticRegression": poly_val_preds,
            "GradientBoostingClassifier": boosted_val_preds,
            "HistGradientBoostingClassifier": hist_val_preds,
        },
    )

    print("=== VALIDATION PREDICTED CLASS DISTRIBUTIONS ===")
    for model_name, model_diagnostics in validation_diagnostics["models"].items():
        print(f"{model_name}: {model_diagnostics['predicted_class_distribution']}")

    save_json(
        RESULTS_DIR / "baseline_metrics.json",
        {
            "model": "LogisticRegression",
            "validation_metrics": baseline_metrics,
            "runtime_seconds": baseline_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "tuned_logistic_metrics.json",
        {
            "model": "TunedLogisticRegression",
            "best_params": tuned_logistic_search.best_params_,
            "best_cv_accuracy": float(tuned_logistic_search.best_score_),
            "validation_metrics": tuned_logistic_metrics,
            "runtime_seconds": tuned_logistic_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "random_forest_metrics.json",
        {
            "model": "RandomForestClassifier",
            "best_params": rf_search.best_params_,
            "best_cv_f1_weighted": float(rf_search.best_score_),
            "validation_metrics": rf_metrics,
            "runtime_seconds": rf_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "extra_trees_metrics.json",
        {
            "model": "ExtraTreesClassifier",
            "best_params": extra_trees_search.best_params_,
            "best_cv_accuracy": float(extra_trees_search.best_score_),
            "validation_metrics": extra_trees_metrics,
            "runtime_seconds": extra_trees_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "polynomial_logistic_metrics.json",
        {
            "model": "PolynomialLogisticRegression",
            "best_params": poly_search.best_params_,
            "best_cv_f1_weighted": float(poly_search.best_score_),
            "validation_metrics": poly_metrics,
            "runtime_seconds": poly_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "boosted_tree_metrics.json",
        {
            "model": "GradientBoostingClassifier",
            "best_params": boosted_search.best_params_,
            "best_cv_f1_weighted": float(boosted_search.best_score_),
            "validation_metrics": boosted_metrics,
            "runtime_seconds": boosted_runtime_seconds,
        },
    )
    save_json(
        RESULTS_DIR / "hist_gradient_boosting_metrics.json",
        {
            "model": "HistGradientBoostingClassifier",
            "best_params": hist_search.best_params_,
            "best_cv_accuracy": float(hist_search.best_score_),
            "validation_metrics": hist_metrics,
            "runtime_seconds": hist_runtime_seconds,
        },
    )
    save_json(RESULTS_DIR / "validation_diagnostics.json", validation_diagnostics)

    log_row = pd.DataFrame(
        [
            {
                "experiment_id": 1,
                "model": "LogisticRegression",
                "target": TARGET_COL,
                "validation_metric": "accuracy",
                "validation_accuracy": baseline_metrics["accuracy"],
                "validation_f1_weighted": baseline_metrics["f1_weighted"],
                "validation_precision": baseline_metrics["precision_weighted"],
                "validation_recall": baseline_metrics["recall_weighted"],
                "runtime_seconds": baseline_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": "Week 2 reproducible baseline run",
            },
            {
                "experiment_id": 2,
                "model": "TunedLogisticRegression",
                "target": TARGET_COL,
                "validation_metric": "accuracy",
                "validation_accuracy": tuned_logistic_metrics["accuracy"],
                "validation_f1_weighted": tuned_logistic_metrics["f1_weighted"],
                "validation_precision": tuned_logistic_metrics["precision_weighted"],
                "validation_recall": tuned_logistic_metrics["recall_weighted"],
                "runtime_seconds": tuned_logistic_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": (
                    "Seasonality and state temperature features; "
                    f"GridSearchCV best params: {tuned_logistic_search.best_params_}"
                ),
            },
            {
                "experiment_id": 3,
                "model": "RandomForestClassifier",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": rf_metrics["accuracy"],
                "validation_f1_weighted": rf_metrics["f1_weighted"],
                "validation_precision": rf_metrics["precision_weighted"],
                "validation_recall": rf_metrics["recall_weighted"],
                "runtime_seconds": rf_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": f"GridSearchCV best params: {rf_search.best_params_}",
            },
            {
                "experiment_id": 4,
                "model": "ExtraTreesClassifier",
                "target": TARGET_COL,
                "validation_metric": "accuracy",
                "validation_accuracy": extra_trees_metrics["accuracy"],
                "validation_f1_weighted": extra_trees_metrics["f1_weighted"],
                "validation_precision": extra_trees_metrics["precision_weighted"],
                "validation_recall": extra_trees_metrics["recall_weighted"],
                "runtime_seconds": extra_trees_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": (
                    "Seasonality and state temperature features; "
                    f"GridSearchCV best params: {extra_trees_search.best_params_}"
                ),
            },
            {
                "experiment_id": 5,
                "model": "PolynomialLogisticRegression",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": poly_metrics["accuracy"],
                "validation_f1_weighted": poly_metrics["f1_weighted"],
                "validation_precision": poly_metrics["precision_weighted"],
                "validation_recall": poly_metrics["recall_weighted"],
                "runtime_seconds": poly_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": (
                    "Numeric degree-2 polynomial features; "
                    f"GridSearchCV best params: {poly_search.best_params_}"
                ),
            },
            {
                "experiment_id": 6,
                "model": "GradientBoostingClassifier",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": boosted_metrics["accuracy"],
                "validation_f1_weighted": boosted_metrics["f1_weighted"],
                "validation_precision": boosted_metrics["precision_weighted"],
                "validation_recall": boosted_metrics["recall_weighted"],
                "runtime_seconds": boosted_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": f"GridSearchCV best params: {boosted_search.best_params_}",
            },
            {
                "experiment_id": 7,
                "model": "HistGradientBoostingClassifier",
                "target": TARGET_COL,
                "validation_metric": "accuracy",
                "validation_accuracy": hist_metrics["accuracy"],
                "validation_f1_weighted": hist_metrics["f1_weighted"],
                "validation_precision": hist_metrics["precision_weighted"],
                "validation_recall": hist_metrics["recall_weighted"],
                "runtime_seconds": hist_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": (
                    "Seasonality and state temperature features; "
                    f"GridSearchCV best params: {hist_search.best_params_}"
                ),
            },
        ]
    )

    log_path = RESULTS_DIR / "experiment_log.csv"
    if log_path.exists():
        old_log = pd.read_csv(log_path)
        if "experiment_id" in old_log.columns:
            old_log = old_log[~old_log["experiment_id"].isin(log_row["experiment_id"])]
        full_log = pd.concat([old_log, log_row], ignore_index=True)
    else:
        full_log = log_row
    full_log.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()
