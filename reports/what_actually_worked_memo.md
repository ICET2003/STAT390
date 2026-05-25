# What Actually Worked Memo

- `burnout_index_non_weather`: `HistGradientBoostingReg` was best with primary metric 0.7938.
- `burnout_index_weather_augmented`: `HistGradientBoostingReg` was best with primary metric 0.7958.
- `treatment_non_weather`: `HistGradientBoosting` was best with primary metric 0.7962.
- `treatment_weather_augmented`: `Logistic_C0.1` was best with primary metric 0.7963.

Weather improves a run only if the best weather-augmented primary metric exceeds the best non-weather primary metric for the same target.
