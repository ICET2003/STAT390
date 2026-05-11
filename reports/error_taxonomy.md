# Error Taxonomy

## Taxonomy Categories

| Error Type | Definition | Evidence in Current Validation Results | Impact |
|---|---|---|---|
| Central-class collapse | Model predicts class `3` for most records regardless of true class | Every model predicts class `3` for 390 to 394 of 450 validation records | Inflates class `3` recall while suppressing minority-class recall |
| Low-rating blindness | Model fails to identify rating `1` | Class `1` recall is 0.0000 for all models | The system misses the most severe burnout-risk rating |
| High-rating blindness | Model fails to identify rating `5` | Class `5` recall is 0.0000 for logistic, random forest, and polynomial logistic; gradient boosting recall is only 0.0227 | The system cannot reliably distinguish high-performing/low-risk employees |
| Adjacent-class compression | Ratings `2` and `4` are absorbed into class `3` | For logistic regression, 62 of 82 true class `2` records and 56 of 72 true class `4` records are predicted as `3` | The model treats near-middle ratings as neutral |
| Minority prediction scarcity | The model rarely emits classes `1`, `4`, or `5` | Logistic regression predicts class `1` only 2 times, class `4` 17 times, and class `5` 4 times | Precision/recall estimates for edge classes are unstable and weak |
| Runtime inefficiency without quality gain | More complex models cost more time without improving validation quality | Gradient boosting takes 31.38 seconds and random forest takes 21.11 seconds, while logistic regression takes 0.11 seconds | Extra complexity is not justified by current validation results |

## Confusion Pattern Snapshot

| Model | Correct / 450 | Predicted Class 3 Count | True Non-3 Predicted as 3 | Class 1 Recall | Class 5 Recall |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 222 | 390 | 182 | 0.0000 | 0.0000 |
| RandomForestClassifier | 220 | 391 | 187 | 0.0000 | 0.0000 |
| PolynomialLogisticRegression | 222 | 392 | 183 | 0.0000 | 0.0000 |
| GradientBoostingClassifier | 222 | 394 | 187 | 0.0000 | 0.0227 |

## Primary Error Mechanism

The dominant failure is not random noise across all labels. It is systematic compression toward the middle rating. The models learn that class `3` is the safest prediction because it is the largest validation class, and the current feature set does not give enough separable signal to recover the edge ratings.
