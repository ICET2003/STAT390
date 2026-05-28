# Final Results Table

Primary metrics are weighted F1 for `sought_treatment` classification and R-squared for `burnout_index` regression.

## Controlled Experiment Results

| Target | Feature Set | Baseline Model | Baseline Metric | Best Model | Best Metric | Improvement |
|---|---|---|---:|---|---:|---:|
| `sought_treatment` | Non-weather | DummyMostFrequent | 0.3915 weighted F1 | HistGradientBoosting | 0.7962 weighted F1 | +0.4047 |
| `sought_treatment` | Weather-augmented | DummyMostFrequent | 0.3915 weighted F1 | Logistic_C0.1 | 0.7963 weighted F1 | +0.4048 |
| `burnout_index` | Non-weather | DummyMean | -0.0001 R-squared | HistGradientBoostingReg | 0.7938 R-squared | +0.7939 |
| `burnout_index` | Weather-augmented | DummyMean | -0.0001 R-squared | HistGradientBoostingReg | 0.7958 R-squared | +0.7959 |

## Focused Improvement Results

| Target | Feature Set | Best Focused Model | Primary Metric | Secondary Metrics | Interpretation |
|---|---|---|---:|---|---|
| `sought_treatment` | Non-weather | HistGB_lr003_leaf15_l2 | 0.7959 | Weighted F1 0.7959 | Did not improve beyond the controlled best. |
| `sought_treatment` | Weather-augmented | Logistic_C0.03_balanced | 0.7897 | Weighted F1 0.7897 | Did not improve beyond the controlled best. |
| `burnout_index` | Non-weather | HistGBReg_lr002_leaf63_l2 | 0.7944 | RMSE 0.8820, MAE 0.7072 | Slightly improves the controlled non-weather result. |
| `burnout_index` | Weather-augmented | HistGBReg_lr002_leaf63_l2 | 0.7962 | RMSE 0.8780, MAE 0.7045 | Best overall burnout-index result. |

## Main Conclusion

Weather variables provide weak incremental predictive value in this project. The weather-augmented burnout-index model is the best overall model, but the gain over the non-weather model is small. Treatment prediction shows almost no improvement from adding weather. Variable-importance analysis suggests that work, sleep, and health variables dominate the predictions, while weather features such as wind gust, pressure, and room temperature contribute smaller signals.

For `sought_treatment`, the best models improve strongly over the dummy baseline by about +0.405 weighted F1. Weather does not materially improve the final treatment metric, but it has a clearer variable-importance signal than it does for the burnout index: `wind_gust` is the third most important feature in the weather-augmented treatment model, and `pressure_hpa` also appears in the top ten.
