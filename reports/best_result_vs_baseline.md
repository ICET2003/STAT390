# Best Result vs. Baseline

## Baseline

| Model | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Runtime sec |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.4933 | 0.3719 | 0.3283 | 0.4933 | 0.11 |

## Best Completed Result

The best completed result is still the baseline logistic regression.

| Comparison Model | Accuracy Delta vs Baseline | Weighted F1 Delta vs Baseline | Runtime Delta vs Baseline | Result |
|---|---:|---:|---:|---|
| RandomForestClassifier | -0.0044 | -0.0017 | +21.00 sec | Worse |
| PolynomialLogisticRegression | 0.0000 | -0.0022 | +3.06 sec | Tie on accuracy, worse F1 |
| GradientBoostingClassifier | 0.0000 | -0.0008 | +31.27 sec | Tie on accuracy, worse F1 |

## Decision

The baseline should remain the best current model. None of the completed alternatives improves validation accuracy, and none improves weighted F1.

## Important Caveat

The seasonality and state temperature feature run was stopped before completion, so it cannot be counted as a valid comparison yet.
