# Focused Model Improvement

This pass tested regularized logistic, histogram-gradient-boosting, ridge, and MLP variants using the same validation split policy as the controlled experiment.

## Best Focused Candidates

| run_id | task_type | model | primary_metric | accuracy | f1_weighted | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burnout_index_non_weather | regression | HistGBReg_lr002_leaf63_l2 | 0.7944 | nan | nan | 0.8820 | 0.7072 | 0.7944 |
| burnout_index_weather_augmented | regression | HistGBReg_lr002_leaf63_l2 | 0.7962 | nan | nan | 0.8780 | 0.7045 | 0.7962 |
| treatment_non_weather | classification | HistGB_lr003_leaf15_l2 | 0.7959 | 0.7959 | 0.7959 | nan | nan | nan |
| treatment_weather_augmented | classification | Logistic_C0.03_balanced | 0.7897 | 0.7891 | 0.7897 | nan | nan | nan |
