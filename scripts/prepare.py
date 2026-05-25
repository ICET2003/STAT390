"""Frozen data preparation for the employee rating project.

This file should stay stable during model experimentation. It owns the raw-data
load, deterministic feature engineering, deterministic train/validation/test
split, and prepared CSV outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/employee_data.csv")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42
TARGET_COL = "Current Employee Rating"

DROP_COLS = [
    "EmpID",
    "FirstName",
    "LastName",
    "StartDate",
    "ExitDate",
    "Title",
    "Supervisor",
    "ADEmail",
    "BusinessUnit",
    "EmployeeStatus",
    "TerminationType",
    "TerminationDescription",
    "DOB",
    "DateofHire",
    "Performance Score",
]

STATE_AVG_TEMP_F = {
    "AL": 62.8,
    "AK": 26.6,
    "AZ": 60.3,
    "AR": 60.4,
    "CA": 59.4,
    "CO": 45.1,
    "CT": 49.0,
    "DE": 55.3,
    "FL": 70.7,
    "GA": 63.5,
    "HI": 70.0,
    "ID": 44.4,
    "IL": 51.8,
    "IN": 51.7,
    "IA": 47.8,
    "KS": 54.3,
    "KY": 55.6,
    "LA": 66.4,
    "ME": 41.0,
    "MD": 54.2,
    "MA": 47.9,
    "MI": 44.4,
    "MN": 41.2,
    "MS": 63.4,
    "MO": 54.5,
    "MT": 42.7,
    "NE": 48.8,
    "NV": 49.9,
    "NH": 43.8,
    "NJ": 52.7,
    "NM": 53.4,
    "NY": 45.4,
    "NC": 59.0,
    "ND": 40.4,
    "OH": 50.7,
    "OK": 59.6,
    "OR": 48.4,
    "PA": 48.8,
    "RI": 50.1,
    "SC": 62.4,
    "SD": 45.2,
    "TN": 57.6,
    "TX": 64.8,
    "UT": 48.6,
    "VT": 42.9,
    "VA": 55.1,
    "WA": 48.3,
    "WV": 51.8,
    "WI": 43.1,
    "WY": 42.0,
    "DC": 58.2,
}

STATE_REGION = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "DC": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["DOB", "DateofHire", "StartDate", "ExitDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reference_date = pd.Timestamp("2025-01-01")

    if "DOB" in df.columns:
        df["Age_Years"] = ((reference_date - df["DOB"]).dt.days / 365.25).round(2)

    if "DateofHire" in df.columns:
        df["Tenure_Days"] = (reference_date - df["DateofHire"]).dt.days

    if "EmployeeStatus" in df.columns:
        df["Is_Active"] = (
            df["EmployeeStatus"].astype(str).str.lower() == "active"
        ).astype(int)

    for date_col in ["StartDate", "DateofHire"]:
        if date_col in df.columns:
            prefix = date_col.replace("Date", "")
            month = df[date_col].dt.month
            quarter = df[date_col].dt.quarter
            df[f"{prefix}_Month"] = month
            df[f"{prefix}_Quarter"] = quarter
            df[f"{prefix}_Year"] = df[date_col].dt.year
            df[f"{prefix}_Month_Sin"] = np.sin(2 * np.pi * month / 12)
            df[f"{prefix}_Month_Cos"] = np.cos(2 * np.pi * month / 12)
            df[f"{prefix}_Season"] = month.map(month_to_season)

    if "State" in df.columns:
        state = df["State"].astype(str).str.upper().str.strip()
        avg_temp = state.map(STATE_AVG_TEMP_F)
        df["State_Avg_Temp_F"] = avg_temp
        df["State_Avg_Temp_C"] = (avg_temp - 32) * 5 / 9
        df["State_Climate_Region"] = state.map(STATE_REGION).fillna("Unknown")
        df["State_Temp_Band"] = pd.cut(
            avg_temp,
            bins=[-np.inf, 45, 55, 65, np.inf],
            labels=["Cold", "Cool", "Mild", "Warm"],
        ).astype("object")

    return df


def month_to_season(month: float) -> str | None:
    if pd.isna(month):
        return None
    month = int(month)
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"


def safe_drop_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=existing_cols)


def prepare_data(
    data_path: Path = DATA_PATH,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = parse_dates(df)
    df = engineer_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    df = df[df[TARGET_COL].notna()].copy()
    y = df[TARGET_COL]
    X = safe_drop_columns(df, DROP_COLS + [TARGET_COL])

    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1764705882,
        random_state=RANDOM_STATE,
    )

    X_train.to_csv(processed_dir / "X_train.csv")
    y_train.to_csv(processed_dir / "y_train.csv")
    X_val.to_csv(processed_dir / "X_val.csv")
    y_val.to_csv(processed_dir / "y_val.csv")

    split_info = {
        "random_state": RANDOM_STATE,
        "target_column": TARGET_COL,
        "feature_revision": "seasonality_and_state_temperature_v1",
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_set_policy": "Locked and not used during search phase.",
    }
    with open(results_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    test_indices = pd.DataFrame({"test_index": X_test.index})
    test_indices.to_csv(results_dir / "locked_test_indices.csv", index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_prepared_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(processed_dir / "X_train.csv", index_col=0)
    X_val = pd.read_csv(processed_dir / "X_val.csv", index_col=0)
    y_train = pd.read_csv(processed_dir / "y_train.csv", index_col=0).iloc[:, 0]
    y_val = pd.read_csv(processed_dir / "y_val.csv", index_col=0).iloc[:, 0]
    return X_train, X_val, y_train, y_val


def prepared_files_exist(processed_dir: Path = PROCESSED_DIR) -> bool:
    required_files = ["X_train.csv", "X_val.csv", "y_train.csv", "y_val.csv"]
    return all((processed_dir / filename).exists() for filename in required_files)


def main() -> None:
    X_train, X_val, X_test, _, _, _ = prepare_data()
    print("Prepared deterministic train/validation/test split.")
    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Locked test rows: {len(X_test)}")


if __name__ == "__main__":
    main()
