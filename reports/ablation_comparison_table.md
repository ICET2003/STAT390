# Ablation and Comparison Table

This table compares the best non-weather run against the best weather-augmented run for each target.

## Controlled Experiment Results

| Target | Task | Non-Weather Best Model | Non-Weather Metric | Weather Best Model | Weather Metric | Weather Delta | Result |
|---|---|---|---:|---|---:|---:|---|
| `sought_treatment` | Classification | HistGradientBoosting | 0.7962 weighted F1 | Logistic_C0.1 | 0.7963 weighted F1 | +0.0002 | Tiny improvement |
| `burnout_index` | Regression | HistGradientBoostingReg | 0.7938 R-squared | HistGradientBoostingReg | 0.7958 R-squared | +0.0020 | Small improvement |

## Focused Improvement Results

This follow-up pass tested a smaller set of tuned logistic, histogram-gradient-boosting, ridge, and neural-network candidates while keeping the original feature sets.

| Target | Task | Non-Weather Best Model | Non-Weather Metric | Weather Best Model | Weather Metric | Weather Delta | Result |
|---|---|---|---:|---|---:|---:|---|
| `sought_treatment` | Classification | HistGB_lr003_leaf15_l2 | 0.7959 weighted F1 | Logistic_C0.03_balanced | 0.7897 weighted F1 | -0.0063 | No improvement over the controlled best |
| `burnout_index` | Regression | HistGBReg_lr002_leaf63_l2 | 0.7944 R-squared | HistGBReg_lr002_leaf63_l2 | 0.7962 R-squared | +0.0019 | Small improvement |

## Interpretation

Weather does not dramatically change predictive performance. The weather-augmented controlled runs are slightly better on both targets, but the improvement is small enough that it should be described cautiously. The focused improvement pass confirms that burnout prediction can be improved slightly with tuned histogram gradient boosting, while treatment prediction does not improve beyond the original controlled best.

The stronger final claim is:

> Weather variables provide weak incremental predictive value in the current validation experiments, but the effect is small and limited by coarse weather matching and by stronger work, sleep, and health predictors.

## Current Best Runs

| Run | Best Model | Primary Metric | Features | Validation Rows |
|---|---|---:|---:|---:|
| `treatment_non_weather` | HistGradientBoosting | 0.7962 | 25 | 147 |
| `treatment_weather_augmented` | Logistic_C0.1 | 0.7963 | 44 | 147 |
| `burnout_index_non_weather` | HistGradientBoostingReg | 0.7938 | 21 | 20000 |
| `burnout_index_weather_augmented` | HistGradientBoostingReg | 0.7958 | 23 | 20000 |

## Current Best Focused Burnout Runs

| Run | Best Model | Primary Metric | Interpretation |
|---|---|---:|---|
| `burnout_index_non_weather` | HistGBReg_lr002_leaf63_l2 | 0.7944 | Best non-weather focused model |
| `burnout_index_weather_augmented` | HistGBReg_lr002_leaf63_l2 | 0.7962 | Best weather-focused model |

## Recommended Next Ablations

- Use historical weather aligned to the survey year instead of current OpenWeatherMap weather.
- Compare state-level weather against region-only features to test whether weather is adding information beyond geography.
- Remove location fields such as `state_code` when testing weather, because location can partially absorb weather effects.
- Add repeated random splits or cross-validation to test whether the small weather improvement is stable.
