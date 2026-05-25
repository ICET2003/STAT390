# Best Result vs. Baseline

| run_id | baseline_model | best_model | baseline_primary | best_primary | delta |
| --- | --- | --- | --- | --- | --- |
| burnout_index_non_weather | DummyMean | HistGradientBoostingReg | -0.0001 | 0.7938 | 0.7939 |
| burnout_index_weather_augmented | DummyMean | HistGradientBoostingReg | -0.0001 | 0.7958 | 0.7959 |
| treatment_non_weather | DummyMostFrequent | HistGradientBoosting | 0.3915 | 0.7962 | 0.4047 |
| treatment_weather_augmented | DummyMostFrequent | Logistic_C0.1 | 0.3915 | 0.7963 | 0.4048 |

## Decision Rule

Keep the best model only when it improves the run's primary validation metric over the dummy baseline.
