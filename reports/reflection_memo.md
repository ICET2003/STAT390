# Reflection Memo

## What Changed During the Project

The project began as a broad weather-and-worker-burnout prediction task. The main challenge was defining a usable burnout outcome. I created a PCA-based `burnout_index` from non-weather stress, sleep, and work indicators so that weather could later be tested as an explanatory input rather than being built into the target.

The project then expanded into a controlled experiment design with paired non-weather and weather-augmented runs for two targets: `sought_treatment` and `burnout_index`. I added a full experiment runner, historical logging, result matrices, metric plots, error reports, neural-network baselines, focused improvement experiments, and permutation-importance reports.

## What Worked

The most reliable models were histogram gradient boosting and regularized logistic regression. For `burnout_index`, histogram gradient boosting achieved the best results, with the focused weather-augmented model reaching R-squared 0.7962. For `sought_treatment`, the best weather and non-weather runs were nearly tied at about 0.796 weighted F1.

The strongest predictors were not weather variables. For treatment prediction, family history, work interference, care options, and mental-health consequences were more important. For burnout-index prediction, sleep disorder risk, occupation, day type, REM percentage, mental-health condition, and BMI dominated.

## What Did Not Work

Weather did not produce a large predictive improvement. This does not mean weather has no relationship to mental health or performance. It means that, in this dataset and modeling setup, weather added little predictive power after stronger individual, work, sleep, and health variables were already included.

The likely reasons are data granularity and target construction. Weather was merged at a broad state level or represented through room temperature/season fields, while the outcome was constructed from variables much closer to burnout. Past research often studies more direct weather exposure, repeated daily measurements, seasonal changes, sunlight duration, or causal effects. This project mainly tested incremental prediction.

## Error Taxonomy Reflection

The experiment pipeline tracked failures separately from completed model runs. Errors were categorized as data/preprocessing failures, model-fit failures, metric/evaluation failures, and report-generation failures. The latest full experiment completed 88 models with zero model failures. Non-fatal sklearn or joblib warnings were not counted as crashes because they did not prevent metrics from being written.

## Final Interpretation

The final conclusion should be cautious: weather variables show weak incremental predictive value, especially for burnout-index regression, but they are not major predictors in this project. The strongest evidence is that burnout and treatment outcomes are better explained by direct work, sleep, and health variables. Weather may still matter indirectly, but the current dataset and validation design do not support a strong claim that weather meaningfully improves prediction.
