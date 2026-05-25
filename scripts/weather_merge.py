"""Merge survey or employee observations with state-level weather features.

This follows the reference project design: clean the mental-health survey,
standardize state codes, and join weather observations by state. If a
state-level weather CSV is not available yet, the script uses documented
state-average temperature as a minimal fallback so the merge remains runnable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare import STATE_AVG_TEMP_F, STATE_REGION


DEFAULT_SURVEY_PATH = Path("data/raw/survey.csv")
DEFAULT_EMPLOYEE_PATH = Path("data/raw/employee_data.csv")
DEFAULT_WEATHER_PATH = Path("data/weather/state_weather.csv")
DEFAULT_SURVEY_OUTPUT_PATH = Path("data/processed/survey_weather_merged.csv")
DEFAULT_EMPLOYEE_OUTPUT_PATH = Path("data/processed/employee_weather_merged.csv")
DEFAULT_METADATA_PATH = Path("results/weather_merge_metadata.json")

WEATHER_FEATURE_CANDIDATES = [
    "temperature_f",
    "temperature_c",
    "feels_like_f",
    "temperature_min_f",
    "temperature_max_f",
    "humidity",
    "pressure_hpa",
    "visibility_m",
    "cloud_cover",
    "precipitation",
    "heat_index",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "rain_1h",
    "rain_3h",
    "snow_1h",
    "snow_3h",
    "sunlight_hours",
    "weather_main",
    "weather_description",
    "weather_code",
    "timezone_offset_seconds",
    "extreme_weather_indicator",
    "lagged_heat_exposure",
]

US_STATE_ABBREV = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def normalize_column_name(column: str) -> str:
    return column.strip().lstrip("\ufeff").lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: normalize_column_name(column) for column in df.columns})


def clean_state_code(value) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()
    if value.upper() in {"", "NA", "NAN", "NONE"}:
        return None
    if len(value) == 2:
        return value.upper()

    return US_STATE_ABBREV.get(value.title())


def clean_gender(value) -> str:
    text = str(value).strip().lower()
    if text in {"male", "m", "man", "cis male", "male-ish", "maile"}:
        return "Male"
    if text in {"female", "f", "woman", "cis female", "femake"}:
        return "Female"
    return "Other/Unknown"


def clean_survey_data(path: Path = DEFAULT_SURVEY_PATH) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))

    if "country" in df.columns:
        df = df[df["country"].astype(str).str.strip().eq("United States")].copy()

    if "state" not in df.columns:
        raise ValueError("Survey data must include a state column for weather merging.")

    df["state_code"] = df["state"].map(clean_state_code)
    df = df[df["state_code"].notna()].copy()

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df[df["age"].between(18, 100)].copy()

    if "gender" in df.columns:
        df["gender_clean"] = df["gender"].map(clean_gender)

    if "treatment" in df.columns:
        df["sought_treatment"] = (
            df["treatment"].astype(str).str.strip().str.lower().eq("yes").astype(int)
        )

    return df


def clean_employee_data(path: Path = DEFAULT_EMPLOYEE_PATH) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))

    if "state" not in df.columns:
        raise ValueError("Employee data must include a state column for weather merging.")

    df["state_code"] = df["state"].map(clean_state_code)
    df = df[df["state_code"].notna()].copy()
    return df


def build_fallback_weather_data() -> pd.DataFrame:
    weather = pd.DataFrame(
        {
            "state_code": list(STATE_AVG_TEMP_F.keys()),
            "temperature_f": list(STATE_AVG_TEMP_F.values()),
        }
    )
    weather["temperature_c"] = (weather["temperature_f"] - 32) * 5 / 9
    weather["climate_region"] = weather["state_code"].map(STATE_REGION).fillna("Unknown")
    weather["temperature_band"] = pd.cut(
        weather["temperature_f"],
        bins=[-float("inf"), 45, 55, 65, float("inf")],
        labels=["Cold", "Cool", "Mild", "Warm"],
    ).astype("object")
    return weather


def load_weather_data(path: Path = DEFAULT_WEATHER_PATH) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return build_fallback_weather_data(), "state_average_temperature_fallback"

    weather = normalize_columns(pd.read_csv(path))

    if "state_code" not in weather.columns:
        if "state" in weather.columns:
            weather["state_code"] = weather["state"].map(clean_state_code)
        elif "state_abbrev" in weather.columns:
            weather["state_code"] = weather["state_abbrev"].map(clean_state_code)
        else:
            raise ValueError(
                "Weather data must include state_code, state_abbrev, or state."
            )

    rename_map = {
        "temp": "temperature_f",
        "temperature": "temperature_f",
        "temperature_fahrenheit": "temperature_f",
        "wind": "wind_speed",
        "windspeed": "wind_speed",
        "sunlight_duration": "sunlight_hours",
        "description": "weather_description",
    }
    weather = weather.rename(columns={k: v for k, v in rename_map.items() if k in weather.columns})
    weather["state_code"] = weather["state_code"].map(clean_state_code)
    weather = weather[weather["state_code"].notna()].copy()

    if "temperature_f" in weather.columns and "temperature_c" not in weather.columns:
        weather["temperature_c"] = (pd.to_numeric(weather["temperature_f"], errors="coerce") - 32) * 5 / 9

    keep_columns = ["state_code"] + [
        column for column in WEATHER_FEATURE_CANDIDATES if column in weather.columns
    ]
    keep_columns += [
        column
        for column in [
            "city",
            "latitude",
            "longitude",
            "start_date",
            "end_date",
            "daylight_hours",
            "retrieved_at_utc",
            "climate_region",
            "temperature_band",
        ]
        if column in weather.columns
    ]
    keep_columns = list(dict.fromkeys(keep_columns))

    return weather[keep_columns].drop_duplicates("state_code"), "state_weather_csv"


def merge_with_weather(
    observations: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    if "state_code" not in observations.columns:
        raise ValueError("Observations must contain state_code before weather merge.")

    return observations.merge(weather, on="state_code", how="inner", validate="many_to_one")


def write_metadata(
    metadata_path: Path,
    dataset: str,
    input_path: Path,
    weather_path: Path,
    output_path: Path,
    weather_source: str,
    merged: pd.DataFrame,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    weather_columns = [
        column for column in WEATHER_FEATURE_CANDIDATES if column in merged.columns
    ]
    payload = {
        "dataset": dataset,
        "input_path": str(input_path),
        "weather_path": str(weather_path),
        "weather_source": weather_source,
        "output_path": str(output_path),
        "rows": int(len(merged)),
        "states": int(merged["state_code"].nunique()),
        "weather_columns": weather_columns,
        "join_key": "state_code",
        "reference_design": (
            "State-level weather merged to mental-health or employee observations, "
            "following the Weather-Effects-on-Mental-Health project pattern."
        ),
    }
    with open(metadata_path, "w") as f:
        json.dump(payload, f, indent=2)


def run_weather_merge(
    dataset: str,
    input_path: Path,
    weather_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    if dataset == "survey":
        observations = clean_survey_data(input_path)
    elif dataset == "employee":
        observations = clean_employee_data(input_path)
    else:
        raise ValueError("dataset must be either 'survey' or 'employee'.")

    weather, weather_source = load_weather_data(weather_path)
    merged = merge_with_weather(observations, weather)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    write_metadata(
        metadata_path=metadata_path,
        dataset=dataset,
        input_path=input_path,
        weather_path=weather_path,
        output_path=output_path,
        weather_source=weather_source,
        merged=merged,
    )

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge survey or employee records with state-level weather data."
    )
    parser.add_argument("--dataset", choices=["survey", "employee"], default="survey")
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if input_path is None:
        input_path = DEFAULT_SURVEY_PATH if args.dataset == "survey" else DEFAULT_EMPLOYEE_PATH
    if output_path is None:
        output_path = (
            DEFAULT_SURVEY_OUTPUT_PATH
            if args.dataset == "survey"
            else DEFAULT_EMPLOYEE_OUTPUT_PATH
        )

    merged = run_weather_merge(
        dataset=args.dataset,
        input_path=input_path,
        weather_path=args.weather,
        output_path=output_path,
        metadata_path=args.metadata,
    )
    print(f"Wrote {len(merged)} merged rows to {output_path}")
    print(f"Wrote weather merge metadata to {args.metadata}")


if __name__ == "__main__":
    main()
