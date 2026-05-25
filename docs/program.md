# Revised Project Statement and Agent Strategy

## Revised Project Statement

This project studies whether weather exposure improves prediction of mental-health and burnout-related outcomes. The core research question is:

> Do weather variables add predictive value beyond worker, demographic, and workplace variables?

The project evaluates this question with two targets:

1. `sought_treatment`: a binary mental-health survey outcome indicating whether a respondent sought mental-health treatment.
2. `burnout_index`: a continuous PCA-derived burnout score constructed from non-weather workplace and worker-state indicators.

The preferred final target is `burnout_index` because it is closer to the burnout research goal. The treatment target is kept as a comparison because it connects directly to the mental-health survey and the weather/mental-health reference project.

## Current Research Design

The project uses a controlled ablation design:

1. Train models without weather variables.
2. Train the same family of models with weather variables added.
3. Compare validation performance.

For `sought_treatment`, the primary metric is weighted F1. For `burnout_index`, the primary metric is R-squared.

Weather is not allowed in the construction of the burnout index. This prevents leakage because weather is later tested as an explanatory feature.

## Data Sources

- `data/raw/survey.csv`: mental-health in tech survey.
- `data/raw/sleep_health_dataset.csv`: sleep, stress, work, and health dataset used to construct the PCA burnout index.
- `data/weather/state_weather.csv`: state-capital weather data downloaded through the weather API pipeline.

## Weather Strategy

The project supports two weather sources:

- Open-Meteo historical weather, preferred for survey-period alignment.
- OpenWeatherMap current weather, useful for reproducing the reference notebook style but weaker for historical inference.

Weather is joined at the state level. This is simple and reproducible, but it is also coarse. A future improvement would use ZIP, county, or city-level weather matched to observation dates.

## Burnout Index Strategy

The burnout index is built with PCA in `scripts/burnout_index.py`.

Included burnout indicators are non-weather proxies such as:

- stress score
- work hours
- sleep quality
- sleep duration
- feeling rested
- cognitive performance
- sleep latency
- wake episodes
- weekend sleep difference

Protective variables are reversed before PCA so higher index values consistently represent higher burnout risk.

## Agent Strategy

The agent pipeline is organized around reproducible scripts rather than notebook-only work:

- `scripts/download_weather.py`: download weather data.
- `scripts/weather_merge.py`: clean survey or employee data and merge weather by state.
- `scripts/burnout_index.py`: construct PCA burnout dimensions.
- `scripts/run_full_experiments.py`: run all controlled model comparisons and regenerate reports.
- `scripts/evaluate.py`: shared evaluation metrics.
- `scripts/model.py`: shared preprocessing pipeline.

The current full experiment runs 20 model variants per run across four controlled runs:

- treatment without weather
- treatment with weather
- burnout index without weather
- burnout index with weather

The runner appends all runs to `results/historical_experiment_log.csv` so historical results are preserved instead of overwritten.

## Current Findings

Latest controlled experiment:

| Target | Non-weather best | Weather best | Interpretation |
|---|---:|---:|---|
| `sought_treatment` weighted F1 | 0.7962 | 0.7963 | Weather adds a negligible improvement. |
| `burnout_index` R-squared | 0.7938 | 0.7958 | Weather adds a small improvement. |

The current evidence suggests weather may add some predictive value, but the effect is small under the current state-level design.

## Main Limitation

Weather is currently assigned by state-level capital weather, not by each worker's exact location and date. This limits the strength of any conclusion about weather effects.
