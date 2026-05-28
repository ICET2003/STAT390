# Submission Memo

## Controlled Experiment Description

The experiment was controlled by using a fixed random seed of 42, the same train-validation split policy inside each target task, and identical preprocessing for competing models within a run. Each target was evaluated in paired conditions: non-weather predictors only and weather-augmented predictors. Weather variables were excluded from burnout-index construction to avoid outcome leakage, and PCA source variables were excluded when predicting the burnout index.

## Error Taxonomy

Errors are classified as data/preprocessing failures, model-fit failures, metric/evaluation failures, and report-generation failures. A run is marked failed only when an exception prevents a model from completing and writing metrics. Non-fatal sklearn warnings are recorded as warnings but are not counted as failures.

Latest run status: 88 complete models and 0 failures.
