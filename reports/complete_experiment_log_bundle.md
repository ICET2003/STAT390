# Complete Experiment Log Bundle

## Bundle Contents

- Source log: `results/historical_experiment_log.csv`
- Latest matrix: `reports/experiment_result_matrix.csv`
- Result matrix: `reports/experiment_result_matrix.md`
- Metric plots: `reports/metric_trajectory_plot.svg`, `reports/metric_over_time_plot.svg`
- Keep/discard/crash summary: `reports/keep_discard_crash_summary.md`
- Best result comparison: `reports/best_result_vs_baseline.md`

## Latest Best Models

| run_id | target | task_type | model | primary_metric | accuracy | f1_weighted | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burnout_index_non_weather | burnout_index | regression | HistGradientBoostingReg | 0.7938 | n/a | n/a | 0.8831 | 0.7084 | 0.7938 |
| burnout_index_weather_augmented | burnout_index | regression | HistGradientBoostingReg | 0.7958 | n/a | n/a | 0.8788 | 0.7054 | 0.7958 |
| treatment_non_weather | sought_treatment | classification | HistGradientBoosting | 0.7962 | 0.7959 | 0.7962 | n/a | n/a | n/a |
| treatment_weather_augmented | sought_treatment | classification | Logistic_C0.1 | 0.7963 | 0.7959 | 0.7963 | n/a | n/a | n/a |

## Reproducibility Rules

- Fixed random seed: `42`.
- Latest runs use validation metrics only.
- Historical runs are appended to `results/historical_experiment_log.csv`.
