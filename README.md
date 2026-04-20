# Employee Baseline Project 

This repository provides a deterministic, end-to-end baseline for predicting **Current Employee Rating** from the employee dataset. This baseline does not include the **local weather** of each state in USA yet. This is a baseline for predicitng the Burout Rating with 1 being very burnout and 5 being productive.

The final project intends to include the weather dataset into predictive analysis, hoping to calculate the SHAP value of each regressor to prove the importance of environment on burning out. The final model should be more accurated than the baseline model to be considered successful.

## Target
- **Outcome variable:** `Current Employee Rating`
- **Model:** Logistic Regression (Multi-Class) baseline
- **Why this target:** numeric discrete 1 to 5 rating makes accuracy a natural, stable validation metric.

## What is excluded from the baseline
- `Performance Score` is excluded because it is too close to the target and can create leakage.
- `EmpID`, names, and email are excluded because they are identifiers rather than useful generalizable predictors.
- `TerminationDescription` is excluded in the first baseline because it is sparse free text and can make the first evaluator unstable.

## Locked evaluation rule
During the search phase, the agent may only evaluate on the **validation set (15% of dataset)**.
The **final test set (15% of dataset)** is saved and must not be accessed until the very end of the project.

## Deterministic split
- Train: 70%
- Validation: 15%
- Test: 15%
- Random seed: `42`

The exact split indices are saved under `data/processed/`.
- X_train: Training Data Regressors
- y_train: Dependent Variable of Training Set
- X_val: Validation Data Regressors
- y_val: Dependent Variable of validation set

## Repository structure (Now only baseline file of py exists)
```text
employee_baseline_project/
├── data/
│   ├── employee_data.csv, weather_data.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_val.csv
│       ├── y_train.csv
│       └── y_val.csv
├── results/
│   ├── baseline_metrics.json
│   └── experiment_log.csv
|   ├── locked_test_indices.csv
│   └── split_info.json
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── experiment_log.py
│   └── run_baseline.py
├── requirements.txt
└── README.md
```

## How to run
From the project root:

```bash
pip install -r requirements.txt
python src/run_baseline.py
```

## Expected behavior
The script will:
1. Load the raw employee dataset
2. Create deterministic engineered features such as age and tenure
3. Split data into train, validation, and test using a fixed seed
4. Train a baseline linear regression model
5. Report validation RMSE, MAE, and R-squared
6. Save results and the first experiment log entry

## Output files
- `results/baseline_metrics.json`: stable evaluator output
- `results/experiment_log.csv`: experiment tracking
- `data/processed/split_plan.json`: locked test set policy

## Notes for the TA
This baseline is designed to be run with a single command and a fixed random seed.
No access to the final test set is needed now.
