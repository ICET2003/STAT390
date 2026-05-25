"""Construct latent burnout dimensions with PCA.

The burnout index is built only from workplace and worker-state indicators.
Weather variables are intentionally excluded because they are later used as
explanatory variables in weather-augmented models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_PATH = Path("data/raw/sleep_health_dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/burnout_index.csv")
DEFAULT_METADATA_PATH = Path("results/burnout_index_pca_metadata.json")

RANDOM_STATE = 42

BURNOUT_INDICATORS = {
    "stress_score": {
        "aliases": ["stress_score", "stress level", "stress_level"],
        "direction": 1,
        "concept": "stress",
    },
    "work_hours_that_day": {
        "aliases": ["work_hours_that_day", "overtime hours", "overtime_hours"],
        "direction": 1,
        "concept": "workload",
    },
    "sleep_quality_score": {
        "aliases": ["sleep_quality_score", "fatigue"],
        "direction": -1,
        "concept": "fatigue_proxy",
    },
    "sleep_duration_hrs": {
        "aliases": ["sleep_duration_hrs"],
        "direction": -1,
        "concept": "rest_deficit_proxy",
    },
    "felt_rested": {
        "aliases": ["felt_rested"],
        "direction": -1,
        "concept": "fatigue_proxy",
    },
    "cognitive_performance_score": {
        "aliases": ["cognitive_performance_score", "productivity decline"],
        "direction": -1,
        "concept": "productivity_decline_proxy",
    },
    "sleep_latency_mins": {
        "aliases": ["sleep_latency_mins"],
        "direction": 1,
        "concept": "sleep_disturbance_proxy",
    },
    "wake_episodes_per_night": {
        "aliases": ["wake_episodes_per_night"],
        "direction": 1,
        "concept": "sleep_disturbance_proxy",
    },
    "weekend_sleep_diff_hrs": {
        "aliases": ["weekend_sleep_diff_hrs"],
        "direction": 1,
        "concept": "sleep_disruption_proxy",
    },
}

WEATHER_EXCLUSION_TERMS = [
    "weather",
    "temperature",
    "humidity",
    "precipitation",
    "heat",
    "wind",
    "sunlight",
    "season",
]


def normalize_column_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def is_weather_column(column: str) -> bool:
    normalized = normalize_column_name(column)
    return any(term in normalized for term in WEATHER_EXCLUSION_TERMS)


def find_indicator_columns(df: pd.DataFrame) -> list[dict]:
    normalized_to_original = {
        normalize_column_name(column).lstrip("\ufeff"): column for column in df.columns
    }
    selected = []

    for canonical_name, config in BURNOUT_INDICATORS.items():
        matched_column = None
        for alias in config["aliases"]:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_original:
                candidate = normalized_to_original[normalized_alias]
                if not is_weather_column(candidate):
                    matched_column = candidate
                    break

        if matched_column is not None:
            selected.append(
                {
                    "canonical_name": canonical_name,
                    "source_column": matched_column,
                    "direction": config["direction"],
                    "concept": config["concept"],
                }
            )

    return selected


def construct_burnout_index(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    n_components: int = 2,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    indicators = find_indicator_columns(df)

    if len(indicators) < 2:
        available = ", ".join(df.columns)
        raise ValueError(
            "PCA burnout index requires at least two non-weather burnout "
            f"indicators. Available columns: {available}"
        )

    feature_frame = pd.DataFrame(index=df.index)
    for indicator in indicators:
        source_column = indicator["source_column"]
        values = pd.to_numeric(df[source_column], errors="coerce")
        feature_frame[indicator["canonical_name"]] = values * indicator["direction"]

    imputed = SimpleImputer(strategy="median").fit_transform(feature_frame)
    scaled = StandardScaler().fit_transform(imputed)

    component_count = min(n_components, len(indicators))
    pca = PCA(n_components=component_count, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)

    result = pd.DataFrame(index=df.index)
    id_columns = [
        column
        for column in ["person_id", "EmpID", "\ufeffEmpID"]
        if column in df.columns
    ]
    for column in id_columns:
        result[column.lstrip("\ufeff")] = df[column]

    for component_index in range(component_count):
        result[f"burnout_pc{component_index + 1}"] = components[:, component_index]

    result["burnout_index"] = result["burnout_pc1"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    metadata = {
        "method": "PCA",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "random_state": RANDOM_STATE,
        "weather_exclusion_terms": WEATHER_EXCLUSION_TERMS,
        "selected_indicators": indicators,
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "components": {
            f"burnout_pc{component_index + 1}": {
                feature: float(loading)
                for feature, loading in zip(
                    feature_frame.columns,
                    pca.components_[component_index],
                    strict=True,
                )
            }
            for component_index in range(component_count)
        },
        "index_definition": "burnout_index is the first principal component.",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct latent burnout dimensions with PCA."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--components", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = construct_burnout_index(
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        n_components=args.components,
    )
    print(f"Wrote {len(result)} burnout scores to {args.output}")
    print(f"Wrote PCA metadata to {args.metadata}")


if __name__ == "__main__":
    main()
