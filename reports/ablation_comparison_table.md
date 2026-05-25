# Ablation and Comparison Table

This table compares the best non-weather run against the best weather-augmented run for each target.

| Target | Task | Non-Weather Best Model | Non-Weather Metric | Weather Best Model | Weather Metric | Weather Delta | Result |
|---|---|---|---:|---|---:|---:|---|
| `sought_treatment` | Classification | HistGradientBoosting | 0.7962 weighted F1 | Logistic_C0.1 | 0.7963 weighted F1 | +0.0002 | Tiny improvement |
| `burnout_index` | Regression | HistGradientBoostingReg | 0.7938 R-squared | HistGradientBoostingReg | 0.7958 R-squared | +0.0020 | Small improvement |

## Interpretation

Weather does not dramatically change predictive performance. The weather-augmented runs are slightly better on both targets, but the improvement is small enough that it should be described cautiously.

The stronger final claim is:

> Weather variables provide a small incremental predictive improvement in the current validation experiments, but the effect is limited by coarse state-level weather matching.

## Current Best Runs

| Run | Best Model | Primary Metric | Features | Validation Rows |
|---|---|---:|---:|---:|
| `treatment_non_weather` | HistGradientBoosting | 0.7962 | 25 | 147 |
| `treatment_weather_augmented` | Logistic_C0.1 | 0.7963 | 44 | 147 |
| `burnout_index_non_weather` | HistGradientBoostingReg | 0.7938 | 21 | 20000 |
| `burnout_index_weather_augmented` | HistGradientBoostingReg | 0.7958 | 23 | 20000 |

## Recommended Next Ablations

- Use historical weather aligned to the survey year instead of current OpenWeatherMap weather.
- Compare state-level weather against region-only features to test whether weather is adding information beyond geography.
- Remove location fields such as `state_code` when testing weather, because location can partially absorb weather effects.
- Add repeated random splits or cross-validation to test whether the small weather improvement is stable.
