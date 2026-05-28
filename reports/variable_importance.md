# Variable Importance

## treatment_non_weather

| feature | importance_mean | importance_std | model |
| --- | --- | --- | --- |
| work_interfere | 0.1266 | 0.0259 | HistGB_lr003_leaf15_l2 |
| care_options | 0.0453 | 0.0190 | HistGB_lr003_leaf15_l2 |
| state | 0.0432 | 0.0160 | HistGB_lr003_leaf15_l2 |
| mental_health_consequence | 0.0362 | 0.0132 | HistGB_lr003_leaf15_l2 |
| family_history | 0.0299 | 0.0117 | HistGB_lr003_leaf15_l2 |
| age | 0.0210 | 0.0145 | HistGB_lr003_leaf15_l2 |
| benefits | 0.0137 | 0.0096 | HistGB_lr003_leaf15_l2 |
| no_employees | 0.0114 | 0.0100 | HistGB_lr003_leaf15_l2 |
| leave | 0.0101 | 0.0073 | HistGB_lr003_leaf15_l2 |
| wellness_program | 0.0090 | 0.0080 | HistGB_lr003_leaf15_l2 |
| seek_help | 0.0088 | 0.0066 | HistGB_lr003_leaf15_l2 |
| gender_clean | 0.0032 | 0.0056 | HistGB_lr003_leaf15_l2 |

## treatment_weather_augmented

| feature | importance_mean | importance_std | model |
| --- | --- | --- | --- |
| family_history | 0.0726 | 0.0075 | Logistic_C0.03_balanced |
| work_interfere | 0.0563 | 0.0129 | Logistic_C0.03_balanced |
| wind_gust | 0.0260 | 0.0078 | Logistic_C0.03_balanced |
| mental_health_consequence | 0.0259 | 0.0078 | Logistic_C0.03_balanced |
| care_options | 0.0198 | 0.0250 | Logistic_C0.03_balanced |
| obs_consequence | 0.0095 | 0.0102 | Logistic_C0.03_balanced |
| benefits | 0.0075 | 0.0116 | Logistic_C0.03_balanced |
| leave | 0.0055 | 0.0138 | Logistic_C0.03_balanced |
| pressure_hpa | 0.0055 | 0.0090 | Logistic_C0.03_balanced |
| gender | 0.0055 | 0.0079 | Logistic_C0.03_balanced |
| remote_work | 0.0041 | 0.0069 | Logistic_C0.03_balanced |
| latitude | 0.0041 | 0.0092 | Logistic_C0.03_balanced |

## burnout_index_non_weather

| feature | importance_mean | importance_std | model |
| --- | --- | --- | --- |
| sleep_disorder_risk | 0.6706 | 0.0048 | HistGBReg_lr002_leaf63_l2 |
| occupation | 0.3658 | 0.0053 | HistGBReg_lr002_leaf63_l2 |
| day_type | 0.1390 | 0.0021 | HistGBReg_lr002_leaf63_l2 |
| rem_percentage | 0.0221 | 0.0023 | HistGBReg_lr002_leaf63_l2 |
| mental_health_condition | 0.0212 | 0.0006 | HistGBReg_lr002_leaf63_l2 |
| bmi | 0.0203 | 0.0009 | HistGBReg_lr002_leaf63_l2 |
| shift_work | 0.0087 | 0.0004 | HistGBReg_lr002_leaf63_l2 |
| deep_sleep_percentage | 0.0028 | 0.0010 | HistGBReg_lr002_leaf63_l2 |
| sleep_aid_used | 0.0017 | 0.0003 | HistGBReg_lr002_leaf63_l2 |
| chronotype | 0.0017 | 0.0003 | HistGBReg_lr002_leaf63_l2 |
| caffeine_mg_before_bed | 0.0017 | 0.0002 | HistGBReg_lr002_leaf63_l2 |
| exercise_day | 0.0015 | 0.0005 | HistGBReg_lr002_leaf63_l2 |

## burnout_index_weather_augmented

| feature | importance_mean | importance_std | model |
| --- | --- | --- | --- |
| sleep_disorder_risk | 0.6699 | 0.0050 | HistGBReg_lr002_leaf63_l2 |
| occupation | 0.3668 | 0.0053 | HistGBReg_lr002_leaf63_l2 |
| day_type | 0.1388 | 0.0018 | HistGBReg_lr002_leaf63_l2 |
| rem_percentage | 0.0223 | 0.0022 | HistGBReg_lr002_leaf63_l2 |
| mental_health_condition | 0.0206 | 0.0008 | HistGBReg_lr002_leaf63_l2 |
| bmi | 0.0203 | 0.0011 | HistGBReg_lr002_leaf63_l2 |
| shift_work | 0.0084 | 0.0003 | HistGBReg_lr002_leaf63_l2 |
| room_temperature_celsius | 0.0032 | 0.0005 | HistGBReg_lr002_leaf63_l2 |
| deep_sleep_percentage | 0.0031 | 0.0010 | HistGBReg_lr002_leaf63_l2 |
| sleep_aid_used | 0.0017 | 0.0003 | HistGBReg_lr002_leaf63_l2 |
| caffeine_mg_before_bed | 0.0016 | 0.0001 | HistGBReg_lr002_leaf63_l2 |
| exercise_day | 0.0014 | 0.0004 | HistGBReg_lr002_leaf63_l2 |

