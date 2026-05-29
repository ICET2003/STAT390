# Final Results Table

**Name:** IceT Thaewanarumitkul

This project used two main metrics:

- `sought_treatment`: weighted F1, because this is a classification task.
- `burnout_index`: R-squared, because this is a regression task.

## Short Answer

The models improved a lot over the simple dummy baselines. Weather features helped only a little overall. Weather was more visible in the `sought_treatment` importance results than in the `burnout_index` importance results.

## Best Controlled Results

| Outcome | No-Weather Best | Weather Best | Weather Change |
|---|---:|---:|---:|
| `sought_treatment` weighted F1 | 0.7962 | 0.7963 | +0.0002 |
| `burnout_index` R-squared | 0.7938 | 0.7958 | +0.0020 |

## Baseline vs. Best Model

| Outcome | Baseline | Best Model | Improvement |
|---|---:|---:|---:|
| `sought_treatment`, no weather | 0.3915 F1 | 0.7962 F1 | +0.4047 F1 |
| `sought_treatment`, weather | 0.3915 F1 | 0.7963 F1 | +0.4048 F1 |
| `burnout_index`, no weather | -0.0001 R-squared | 0.7938 R-squared | +0.7939 R-squared |
| `burnout_index`, weather | -0.0001 R-squared | 0.7958 R-squared | +0.7959 R-squared |

The baseline for `sought_treatment` was `DummyMostFrequent`. The baseline for `burnout_index` was `DummyMean`. The burnout baseline is listed as about zero because its R-squared was `-0.0001`, which means it explained essentially no validation variation.

## Burnout Regression Error Metrics

| Run | Model | RMSE | MAE | R-squared |
|---|---|---:|---:|---:|
| `burnout_index_non_weather` | DummyMean baseline | 1.9449 | 1.5925 | -0.0001 |
| `burnout_index_non_weather` | HistGradientBoostingReg best | 0.8831 | 0.7084 | 0.7938 |
| `burnout_index_weather_augmented` | DummyMean baseline | 1.9449 | 1.5925 | -0.0001 |
| `burnout_index_weather_augmented` | HistGradientBoostingReg best | 0.8788 | 0.7054 | 0.7958 |

RMSE is useful because it shows the prediction error size for the burnout-index score. R-squared is useful because it shows explained validation variation. The best weather-augmented burnout model reduced RMSE from `1.9449` to `0.8788`.

## Best Models

| Run | Best Model | Metric |
|---|---|---:|
| `treatment_non_weather` | HistGradientBoosting | 0.7962 weighted F1 |
| `treatment_weather_augmented` | Logistic_C0.1 | 0.7963 weighted F1 |
| `burnout_index_non_weather` | HistGradientBoostingReg | 0.7938 R-squared |
| `burnout_index_weather_augmented` | HistGradientBoostingReg | 0.7958 R-squared |

## Focused Improvement Results

After the main controlled experiment, I tested a smaller set of tuned models.

| Outcome | No-Weather Best | Weather Best | Takeaway |
|---|---:|---:|---|
| `sought_treatment` | 0.7959 | 0.7897 | Did not beat the controlled best result. |
| `burnout_index` | 0.7944 | 0.7962 | Slightly improved the burnout result. |

The best overall burnout-index result was the focused weather-augmented histogram-gradient-boosting model with R-squared `0.7962`.

## Weather Importance

Weather did not strongly improve the final metrics, but some weather variables appeared in the importance results.

For `sought_treatment`, weather had the clearest signal:

| Feature | Importance |
|---|---:|
| `wind_gust` | 0.0260 |
| `pressure_hpa` | 0.0055 |

For `burnout_index`, the weather signal was weaker:

| Feature | Importance |
|---|---:|
| `room_temperature_celsius` | 0.0032 |

## Final Interpretation

The project found strong predictive improvement over baseline models. However, weather only added a small amount of extra predictive value. The strongest predictors were work, sleep, health, and mental-health variables. Weather may still matter indirectly, especially for treatment prediction through wind and pressure, but the results do not support a strong claim that weather is a major predictor in this dataset.
