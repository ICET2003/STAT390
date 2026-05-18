# Complete Experiment Log Bundle

## Bundle Contents

This bundle summarizes the controlled validation experiments completed so far. The locked test set was not used.

- Source log: `results/experiment_log.csv`
- Metric files: `results/baseline_metrics.json`, `results/random_forest_metrics.json`, `results/polynomial_logistic_metrics.json`, `results/boosted_tree_metrics.json`
- Diagnostics: `results/validation_diagnostics.json`
- Split metadata: `results/split_info.json`
- Locked test indices: `results/locked_test_indices.csv`
- Result matrix: `reports/experiment_result_matrix.md`
- Metric plot: `reports/metric_trajectory_plot.svg`
- Keep/discard/crash summary: `reports/keep_discard_crash_summary.md`
- Best result comparison: `reports/best_result_vs_baseline.md`
- Worked memo: `reports/what_actually_worked_memo.md`

## Completed Experiment Log

| ID | Status | Model | Controlled Change | Validation Accuracy | Weighted F1 | Runtime sec | Decision |
|---:|---|---|---|---:|---:|---:|---|
| 1 | Complete | LogisticRegression | Baseline linear classifier | 0.4933 | 0.3719 | 0.11 | Keep as baseline |
| 2 | Complete | RandomForestClassifier | Tuned nonlinear bagged trees | 0.4889 | 0.3702 | 21.11 | Discard |
| 3 | Complete | PolynomialLogisticRegression | Degree-2 numeric interactions | 0.4933 | 0.3697 | 3.17 | Discard |
| 4 | Complete | GradientBoostingClassifier | Tuned sequential boosted trees | 0.4933 | 0.3711 | 31.38 | Discard for now |
| 5 | Stopped | Expanded tuning plus seasonality/state temperature features | Added feature revision and larger grids | n/a | n/a | n/a | Crash/stopped, not valid evidence |

## Reproducibility Rules

- Use only validation metrics for model search.
- Do not evaluate on the locked test set until final model selection.
- Treat stopped or interrupted runs as invalid until they complete and write metrics.
- Compare future experiments against experiment 1 unless a new baseline is explicitly declared.

## Current Conclusion

No completed experiment beats the baseline in a meaningful way. The baseline logistic regression has the best weighted F1 and ties the best validation accuracy while running much faster than the tuned tree models.
