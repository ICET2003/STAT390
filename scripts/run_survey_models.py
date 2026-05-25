"""Run non-weather and weather-augmented model suites for the survey dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from evaluate import evaluate_classification
from model import RANDOM_STATE, build_preprocessor
from weather_merge import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SURVEY_OUTPUT_PATH,
    DEFAULT_WEATHER_PATH,
    WEATHER_FEATURE_CANDIDATES,
    run_weather_merge,
)


INPUT_PATH = DEFAULT_SURVEY_OUTPUT_PATH
RESULTS_PATH = Path("results/survey_model_comparison.csv")
SUMMARY_PATH = Path("results/survey_model_comparison_summary.json")
TARGET_COL = "sought_treatment"

EXCLUDE_ALWAYS = [
    TARGET_COL,
    "treatment",
    "timestamp",
    "comments",
    "country",
]

WEATHER_COLUMNS = WEATHER_FEATURE_CANDIDATES + [
    "daylight_hours",
    "climate_region",
    "temperature_band",
    "city",
    "latitude",
    "longitude",
    "start_date",
    "end_date",
    "retrieved_at_utc",
]


def build_model_suite() -> dict[str, object]:
    return {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
        "SGDLogistic": SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "LinearSVC": LinearSVC(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=5000,
        ),
        "KNeighbors": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=5,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def select_features(df: pd.DataFrame, include_weather: bool) -> pd.DataFrame:
    excluded = set(EXCLUDE_ALWAYS)
    if not include_weather:
        excluded.update(column for column in WEATHER_COLUMNS if column in df.columns)

    selected_columns = [
        column
        for column in df.columns
        if column not in excluded
        and column.split(".")[0] not in excluded
        and not column.startswith("weather_")
    ]
    features = df[selected_columns].copy()
    return features.dropna(axis=1, how="all")


def train_validation_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_and_score_models(
    X: pd.DataFrame,
    y: pd.Series,
    run_name: str,
) -> tuple[list[dict], dict]:
    X_train, X_val, y_train, y_val = train_validation_split(X, y)
    rows = []
    predictions = {}

    for model_name, estimator in build_model_suite().items():
        start = time.time()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("clf", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        runtime_seconds = time.time() - start
        metrics = evaluate_classification(y_val, y_pred)
        predictions[model_name] = y_pred

        row = {
            "run_name": run_name,
            "model": model_name,
            "n_features": int(X.shape[1]),
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
            "runtime_seconds": runtime_seconds,
            **metrics,
        }
        rows.append(row)
        print(
            f"{run_name} | {model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_weighted']:.4f}, "
            f"runtime={runtime_seconds:.2f}s"
        )

    return rows, {
        "validation_class_distribution": y_val.value_counts().sort_index().to_dict(),
        "feature_columns": X.columns.tolist(),
    }


def run_survey_model_comparison(
    input_path: Path = INPUT_PATH,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    if not input_path.exists():
        run_weather_merge(
            dataset="survey",
            input_path=Path("data/raw/survey.csv"),
            weather_path=DEFAULT_WEATHER_PATH,
            output_path=input_path,
            metadata_path=DEFAULT_METADATA_PATH,
        )

    df = pd.read_csv(input_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{input_path} must contain {TARGET_COL}.")

    y = df[TARGET_COL].astype(int)
    non_weather_X = select_features(df, include_weather=False)
    weather_X = select_features(df, include_weather=True)

    all_rows = []
    summary = {
        "target": TARGET_COL,
        "input_path": str(input_path),
        "random_state": RANDOM_STATE,
        "runs": {},
    }

    rows, run_summary = fit_and_score_models(
        non_weather_X,
        y,
        run_name="non_weather_only",
    )
    all_rows.extend(rows)
    summary["runs"]["non_weather_only"] = run_summary

    rows, run_summary = fit_and_score_models(
        weather_X,
        y,
        run_name="weather_augmented",
    )
    all_rows.extend(rows)
    summary["runs"]["weather_augmented"] = run_summary

    results = pd.DataFrame(all_rows).sort_values(
        ["run_name", "f1_weighted", "accuracy"],
        ascending=[True, False, False],
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    best_by_run = (
        results.sort_values(["run_name", "f1_weighted"], ascending=[True, False])
        .groupby("run_name")
        .head(1)
        .to_dict(orient="records")
    )
    summary["best_by_run"] = best_by_run
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many models with and without weather features."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_survey_model_comparison(
        input_path=args.input,
        results_path=args.results,
        summary_path=args.summary,
    )
    print(f"Wrote {len(results)} model results to {args.results}")
    print(f"Wrote run summary to {args.summary}")


if __name__ == "__main__":
    main()
