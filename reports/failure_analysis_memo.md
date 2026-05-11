# Failure Analysis Memo

## Summary

The controlled experiments do not show a meaningful improvement over the baseline. Validation accuracy stays near 0.49, weighted F1 stays near 0.37, and all four models mostly predict class `3`. The main failure is middle-class collapse: the models avoid predicting the edge ratings even when the true label is `1`, `4`, or `5`.

## What Failed

The experiment set was intended to test whether nonlinear model families or polynomial numeric interactions could improve prediction of `Current Employee Rating`. That did not happen. Logistic regression, polynomial logistic regression, and gradient boosting all reached 0.4933 validation accuracy, while random forest reached 0.4889. Weighted F1 changed by less than 0.003 across all four experiments.

The class-level diagnostics show why aggregate accuracy is misleading. Class `3` has strong recall because the models predict it for almost every row. In contrast, class `1` has zero recall for every model. Class `5` has zero recall for three models and only 0.0227 recall for gradient boosting.

## Likely Causes

1. The target distribution is imbalanced around class `3`, so optimizing standard accuracy or weighted F1 rewards middle-class predictions.
2. The current baseline excludes likely leakage fields correctly, but the remaining features may not contain enough signal to distinguish rating extremes.
3. Treating the target as a nominal multiclass label discards the ordinal structure of ratings `1` through `5`.
4. Grid search improved neither representation nor objective enough to counter the class imbalance.

## Evidence

The strongest evidence is the predicted class distribution. Logistic regression predicts class `3` for 390 of 450 validation rows. Random forest predicts class `3` for 391 rows, polynomial logistic regression for 392 rows, and gradient boosting for 394 rows.

This leads to high class `3` recall but poor edge-class performance. Logistic regression correctly classifies 208 of 215 true class `3` examples, but it correctly classifies 0 of 37 true class `1` examples and 0 of 44 true class `5` examples.

## Recommendation

The next experiment should focus on the failure mode directly instead of adding another generic classifier. Strong next steps are:

1. Add class imbalance handling to the primary objective, such as macro F1 model selection or explicit sample weights.
2. Evaluate ordinal-aware approaches, such as predicting rating as an ordered outcome or using regression followed by calibrated rating bins.
3. Add weather/environment features only after preserving the same locked split, so the new feature family can be tested as the controlled change.
4. Track macro F1 and per-class recall in the experiment matrix, because accuracy and weighted F1 hide the edge-rating failures.

## Decision

Do not select the slower tree-search models as final candidates yet. The baseline logistic regression is faster and performs about the same on validation. The modeling priority should shift from model complexity to class-balance handling, ordinal target treatment, and stronger feature signal.
