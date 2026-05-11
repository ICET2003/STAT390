"""Run validation experiments using frozen prepared data and editable models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from model import (
    build_boosted_tree_search_space,
    build_logistic_regression,
    build_polynomial_logistic_search_space,
    build_random_forest_search_space,
    build_validation_diagnostics,
    evaluate_classification,
)
from prepare import (
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COL,
    load_prepared_data,
    prepare_data,
    prepared_files_exist,
)


def print_metrics(title: str, metrics: dict, runtime_seconds: float) -> None:
    print(f"=== {title} VALIDATION RESULTS ===")
    print(f"Validation ACC: {metrics['accuracy']:.4f}")
    print(f"Validation F1:  {metrics['f1_weighted']:.4f}")
    print(f"Validation Precision:  {metrics['precision_weighted']:.4f}")
    print(f"Validation Recall: {metrics['recall_weighted']:.4f}")
    print(f"Runtime (seconds): {runtime_seconds:.4f}")


def run_grid_search(name: str, pipeline, param_grid: dict, X_train, y_train, X_val, y_val):
    start_time = time.time()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    val_preds = search.predict(X_val)
    metrics = evaluate_classification(y_val, val_preds)
    runtime_seconds = time.time() - start_time

    print(f"=== {name} VALIDATION RESULTS ===")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")
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

    if not prepared_files_exist():
        print("Prepared data files not found. Running frozen preparation first.")
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

    validation_diagnostics = build_validation_diagnostics(
        y_val,
        {
            "LogisticRegression": baseline_val_preds,
            "RandomForestClassifier": rf_val_preds,
            "PolynomialLogisticRegression": poly_val_preds,
            "GradientBoostingClassifier": boosted_val_preds,
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
    save_json(RESULTS_DIR / "validation_diagnostics.json", validation_diagnostics)

    log_row = pd.DataFrame(
        [
            {
                "experiment_id": 1,
                "model": "LogisticRegression",
                "target": TARGET_COL,
                "validation_metric": "accuracy",
                "validation_accuracy": baseline_metrics["accuracy"],
                "validation_precision": baseline_metrics["precision_weighted"],
                "validation_recall": baseline_metrics["recall_weighted"],
                "runtime_seconds": baseline_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": "Week 2 reproducible baseline run",
            },
            {
                "experiment_id": 2,
                "model": "RandomForestClassifier",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": rf_metrics["accuracy"],
                "validation_precision": rf_metrics["precision_weighted"],
                "validation_recall": rf_metrics["recall_weighted"],
                "runtime_seconds": rf_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": f"GridSearchCV best params: {rf_search.best_params_}",
            },
            {
                "experiment_id": 3,
                "model": "PolynomialLogisticRegression",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": poly_metrics["accuracy"],
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
                "experiment_id": 4,
                "model": "GradientBoostingClassifier",
                "target": TARGET_COL,
                "validation_metric": "f1_weighted",
                "validation_accuracy": boosted_metrics["accuracy"],
                "validation_precision": boosted_metrics["precision_weighted"],
                "validation_recall": boosted_metrics["recall_weighted"],
                "runtime_seconds": boosted_runtime_seconds,
                "random_state": RANDOM_STATE,
                "notes": f"GridSearchCV best params: {boosted_search.best_params_}",
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
