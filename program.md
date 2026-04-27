# Program Documentation

## Project Overview

This project builds a deterministic baseline model for predicting `Current Employee Rating` from the employee dataset. The target rating ranges from 1 to 5, where lower values represent higher burnout risk and higher values represent stronger productivity.

The current implementation is contained in `baseline.ipynb`. It loads employee data, creates basic engineered features, performs a fixed train/validation/test split, trains a logistic regression classifier, evaluates on the validation set, and saves reproducible outputs.

## Main Program File

- `baseline.ipynb`: end-to-end baseline notebook

## Input Data

The main input file is:

- `data/employee_data.csv`

The target column is:

- `Current Employee Rating`

## Program Steps

1. Import required packages.
2. Define project configuration values:
   - input data path
   - random seed
   - target column
   - columns to exclude
   - results directory
3. Load the employee dataset.
4. Parse date columns.
5. Engineer additional features:
   - `Age_Years`
   - `Tenure_Days`
   - `Is_Active`
6. Remove rows with missing target values.
7. Separate features `X` from target `y`.
8. Drop identifier, personal information, leakage, and unused columns.
9. Split the data into:
   - 70% training
   - 15% validation
   - 15% locked test
10. Save processed training and validation data.
11. Build a preprocessing pipeline:
   - numeric columns: median imputation and standard scaling
   - categorical columns: most frequent imputation and one-hot encoding
12. Train a logistic regression classification model.
13. Evaluate the model on the validation set.
14. Save metrics, split information, experiment log, and locked test indices.

## Loops Used

The program uses one explicit loop in `parse_dates`:

```python
for col in possible_date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
```

This loop checks each possible date column and only parses it if the column exists in the dataset.

The program also uses one list comprehension in `safe_drop_columns`:

```python
existing_cols = [c for c in cols_to_drop if c in df.columns]
```

This safely filters the drop list so the program only drops columns that are actually present.

## Model

The baseline model is:

- `LogisticRegression`

This is appropriate because the target is a multi-class rating from 1 to 5.

## Validation Metrics

The program calculates:

- accuracy
- weighted F1 score
- weighted precision
- weighted recall

## Output Files

The program writes the following files:

- `data/processed/X_train.csv`
- `data/processed/y_train.csv`
- `data/processed/X_val.csv`
- `data/processed/y_val.csv`
- `results/split_info.json`
- `results/baseline_metrics.json`
- `results/experiment_log.csv`
- `results/locked_test_indices.csv`

## Locked Test Set Policy

The test set is created and saved, but it should not be used during model search or baseline tuning. During development, only the validation set should be used for checking model performance.

## Notes

The date parsing loop currently uses automatic date parsing. Since the dataset contains mixed date formats, it would be safer to specify exact formats for each date column in a future revision.
