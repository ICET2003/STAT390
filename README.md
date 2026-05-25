# Weather and Worker Burnout Research Pipeline

This project studies whether weather exposure improves prediction of worker burnout beyond HR and demographic information. The current implementation contains a reproducible employee-rating baseline; the planned research pipeline extends it into burnout index construction, weather integration, weather-augmented modeling, and heterogeneous effect analysis.

## Research Goal

Quantify how weather exposure affects worker burnout and test whether environmental conditions improve predictive burnout modeling.

Primary comparison:

```text
BurnoutIndex ~ HR + Demographics
BurnoutIndex ~ HR + Demographics + Weather
```

Weather variables must not be used to construct the burnout index. Weather is an explanatory input in later models, so including it inside the outcome would create leakage and circular prediction.

## Stage 1: Burnout Index Construction

Construct a burnout index from burnout-related workplace variables only.

Potential burnout indicators:

- Stress level
- Fatigue
- Overtime hours
- Absenteeism
- Job satisfaction
- Disengagement
- Turnover intention
- Productivity decline

Methodology options:

- Standardized weighted sum
- Principal Component Analysis, implemented in `scripts/burnout_index.py`
- Factor analysis

Correct setup:

```text
BurnoutIndex = f(stress, fatigue, satisfaction, absenteeism)
```

Incorrect setup:

```text
BurnoutIndex = f(stress, fatigue, weather)
```

## Stage 2: Weather Data Integration

Merge worker observations with weather data using available geography.

Potential geographic levels:

- ZIP code
- County
- Metropolitan area
- State

Potential weather variables:

- Temperature
- Humidity
- Precipitation
- Heat index
- Wind speed
- Sunlight duration
- Extreme weather indicators
- Lagged heat exposure

Potential data sources:

- NOAA
- Open-Meteo
- Meteostat
- ERA5
- State-level climate datasets

The current implementation follows the same state-level merge pattern as the `Weather-Effects-on-Mental-Health` reference project: clean the mental-health survey, standardize state codes, then merge weather features by state.

## Stage 3: Baseline Modeling

Predict burnout using HR and demographic variables only.

Potential models:

- Linear regression
- Logistic regression, if the outcome is categorized
- Random forest
- XGBoost or gradient boosting

The current baseline predicts `Current Employee Rating` as a proxy outcome until the final burnout index is available.

## Stage 4: Weather-Augmented Modeling

Add weather variables and compare incremental predictive value.

Metrics:

- R-squared, for regression
- RMSE
- MAE
- Classification accuracy/F1, if categorized
- Feature importance
- SHAP values

## Stage 5: Heterogeneous Effect Analysis

Test whether weather effects differ across worker groups.

Potential subgroup analyses:

- Indoor vs. outdoor workers
- Remote vs. in-person workers
- Income groups
- Regions or climate zones
- Age groups

Potential interaction:

```text
Temperature x OutdoorWorker
```

## Reproducibility Requirements

- Fixed random seed: `42`
- Locked test dataset
- Deterministic train-validation-test split
- Experiment logging for runtime, hyperparameters, metrics, and preprocessing revisions

## Repository Structure

```text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── weather/
├── docs/
├── figures/
├── notebooks/
├── reports/
├── results/
├── scripts/
│   ├── prepare.py
│   ├── burnout_index.py
│   ├── weather_merge.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── run.py
├── requirements.txt
└── README.md
```

## Script Responsibilities

- `scripts/prepare.py`: cleans the employee dataset, engineers deterministic features, creates locked train/validation/test splits, and writes processed CSV files.
- `scripts/burnout_index.py`: planned module for PCA, factor analysis, or weighted burnout score construction.
- `scripts/weather_merge.py`: planned module for geographic weather merges, lagged weather variables, and heat exposure metrics.
- `scripts/model.py`: model and preprocessing definitions.
- `scripts/evaluate.py`: validation metrics, confusion matrices, and diagnostic reports.
- `scripts/visualize.py`: planned module for SHAP plots, feature importance, correlation matrices, and weather-burnout plots.
- `scripts/run.py`: runs the current baseline experiment suite and writes metrics to `results/`.

## How to Run the Current Baseline

Install dependencies:

```bash
pip install -r requirements.txt
```

Build PCA burnout dimensions:

```bash
python scripts/burnout_index.py
```

Default outputs:

- `data/processed/burnout_index.csv`
- `results/burnout_index_pca_metadata.json`

Merge state-level weather data into the mental-health survey:

```bash
python scripts/download_weather.py
python scripts/weather_merge.py
```

The default downloader uses Open-Meteo historical weather because it does not require an API key and can match the 2014 survey period. To use OpenWeatherMap current weather in the style of the reference notebook, set an API key and choose that provider:

```bash
OPENWEATHER_API_KEY=your_key python scripts/download_weather.py --provider openweathermap
python scripts/weather_merge.py
```

By default, this writes:

- `data/weather/state_weather.csv`
- `data/processed/survey_weather_merged.csv`
- `results/weather_download_metadata.json`
- `results/weather_merge_metadata.json`

If `data/weather/state_weather.csv` exists, it will be used as the weather source. Expected columns include `state_code` plus any of `temperature_f`, `humidity`, `wind_speed`, `sunlight_hours`, `precipitation`, `heat_index`, and `weather_description`. If no weather CSV exists yet, the script uses state-average temperature as a documented fallback so the pipeline remains runnable.

Run the survey model comparison with non-weather features first and weather-augmented features second:

```bash
python scripts/run_survey_models.py
```

Default outputs:

- `results/survey_model_comparison.csv`
- `results/survey_model_comparison_summary.json`

Run the full controlled experiment suite for both targets:

```bash
python scripts/run_full_experiments.py
```

This runs 20 model variants for each of these four runs:

- Treatment prediction without weather
- Treatment prediction with weather
- Burnout-index prediction without weather
- Burnout-index prediction with weather

It regenerates the report bundle under `reports/` and appends every completed historical run to `results/historical_experiment_log.csv`.

Run the pipeline from the project root:

```bash
python scripts/run.py
```

The script regenerates processed data, trains validation models, and writes metrics and experiment logs under `results/`. The final test set remains locked during model search.

## Expected Contributions

This project contributes to:

- Workplace analytics
- Labor economics
- Environmental economics
- Climate adaptation research
- Burnout measurement methodology

Potential novel contribution: quantifying whether weather exposure improves burnout prediction and identifying worker groups most vulnerable to environmental conditions.

## Future Extensions

- Panel data methods
- Causal inference
- Fixed effects estimation
- Remote sensing climate exposure
- Occupational heat vulnerability index
- Temporal weather shock analysis
- Firm-level productivity impacts
