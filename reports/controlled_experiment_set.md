# Controlled Experiment Set

## Experimental Control

All experiments use the same frozen validation protocol:

- Dataset: `data/employee_data.csv`
- Target: `Current Employee Rating`
- Random seed: `42`
- Split: 70% train, 15% validation, 15% locked test
- Search rule: only validation results are used during model selection
- Locked test rule: test indices are saved in `results/locked_test_indices.csv` and are not evaluated during the search phase
- Shared preprocessing: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features
- Leakage controls: identifiers, personal fields, employment status leakage fields, and `Performance Score` are excluded

## Experiment Set

| Experiment ID | Model | Controlled Change | Search Space | Selection Metric |
|---:|---|---|---|---|
| 1 | LogisticRegression | Baseline linear classifier | No grid search | Accuracy |
| 2 | RandomForestClassifier | Nonlinear bagged tree model | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | Weighted F1 |
| 3 | PolynomialLogisticRegression | Logistic regression with degree-2 numeric interactions | `C`, `class_weight` | Weighted F1 |
| 4 | GradientBoostingClassifier | Sequential boosted tree model | `n_estimators`, `learning_rate`, `max_depth` | Weighted F1 |

## Best Hyperparameters

| Experiment ID | Model | Best Parameters |
|---:|---|---|
| 1 | LogisticRegression | `max_iter=1000` |
| 2 | RandomForestClassifier | `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1`, `n_estimators=100` |
| 3 | PolynomialLogisticRegression | `C=1.0`, `class_weight=None` |
| 4 | GradientBoostingClassifier | `learning_rate=0.05`, `max_depth=3`, `n_estimators=50` |

## Interpretation Rule

The controlled comparison asks whether changing only the model family or numeric interaction structure improves validation performance under the same split and preprocessing regime. Because validation accuracy is effectively tied across three models and weighted F1 remains around 0.37 for every model, the current evidence does not support a meaningful improvement over the baseline.
