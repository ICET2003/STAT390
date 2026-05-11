"""Frozen data preparation for the employee rating project.

This file should stay stable during model experimentation. It owns the raw-data
load, deterministic feature engineering, deterministic train/validation/test
split, and prepared CSV outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/employee_data.csv")
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

    return df


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
