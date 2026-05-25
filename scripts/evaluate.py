"""Evaluation helpers for model experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def build_validation_diagnostics(y_true: pd.Series, model_predictions: dict) -> dict:
    labels = sorted(y_true.dropna().unique().tolist())
    diagnostics = {
        "true_class_distribution": y_true.value_counts().sort_index().astype(int).to_dict(),
        "models": {},
    }

    for model_name, y_pred in model_predictions.items():
        pred_series = pd.Series(y_pred)
        diagnostics["models"][model_name] = {
            "predicted_class_distribution": pred_series.value_counts()
            .sort_index()
            .astype(int)
            .to_dict(),
            "confusion_matrix_labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }

    return diagnostics
