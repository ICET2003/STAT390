# Final Presentation Slide Content

**Presenter:** IceT Thaewanarumitkul  
**Recommended length:** 6 minutes  
**Project:** Weather, Mental-Health Treatment, and Burnout Prediction

## Slide 1: Title + Claim

**Title:** Weather, Mental-Health Treatment, and Burnout Prediction

**Claim:**  
Weather variables showed some signal, especially for treatment prediction, but they added only weak incremental predictive value after work, sleep, health, and mental-health variables were included.

**Put on slide:**

- IceT Thaewanarumitkul
- STAT 390 Final Project
- Main claim in one sentence
- Optional small subtitle: Predicting `sought_treatment` and `burnout_index`

**Speaker note:**  
My project started from the idea that weather may affect mental health and performance. The final result is more cautious: weather appears in the models, especially for treatment prediction, but it is not the dominant predictor.

## Slide 2: Topic to AutoResearch Formulation

**Topic:**  
Weather and worker burnout / mental-health outcomes.

**AutoResearch contract:**  
Test whether weather variables improve prediction beyond non-weather worker, health, sleep, and workplace variables.

**Research question:**  
Do weather features improve predictive performance for:

- `sought_treatment`
- `burnout_index`

**Put on slide:**

| Original Topic | Formalized Research Question |
|---|---|
| Weather may affect mood, burnout, or performance | Does weather improve prediction after stronger non-weather predictors are already included? |

**Speaker note:**  
The key shift was changing a broad causal question into a measurable prediction task. I was not proving whether weather causes depression. I was testing whether weather improves model performance in this dataset.

## Slide 3: Data and Baseline

**Data sources:**

- Mental-health survey dataset
- Sleep / work / health dataset
- Weather data merged into modeling tables

**Targets and metrics:**

| Target | Task | Metric |
|---|---|---|
| `sought_treatment` | Classification | Weighted F1 |
| `burnout_index` | Regression | R-squared |

**Baselines:**

| Target | Baseline Model | Baseline |
|---|---|---:|
| `sought_treatment` | DummyMostFrequent | 0.3915 weighted F1 |
| `burnout_index` | DummyMean | RMSE 1.9449, R-squared -0.0001 |

**Speaker note:**  
The burnout baseline has RMSE 1.9449 and R-squared approximately zero because a dummy mean regressor explains essentially none of the validation variation. This gives a clear starting point for measuring improvement.

## Slide 4: Loop Design

**Agent loop / pipeline:**

1. Clean and prepare datasets
2. Construct PCA burnout index without weather leakage
3. Merge weather variables
4. Run non-weather and weather-augmented experiments
5. Log all model runs
6. Compare best models to baselines
7. Add neural networks and focused tuning
8. Compute variable importance

**Put on slide:**

```text
Data -> Burnout Index -> Weather Merge -> Model Suite -> Logs -> Reports -> Importance
```

**Important control rules:**

- Fixed random seed: `42`
- Same train-validation split policy
- Same preprocessing pipeline
- Same model families within each comparison
- Weather excluded from burnout-index construction

**Speaker note:**  
The loop was built around reproducible scripts instead of manual notebook-only work. Every experiment was logged, so I could compare results over time instead of only keeping the best run.

## Slide 5: Experiment Trace

**Key experiment stages:**

| Stage | What I Tried | What It Revealed |
|---|---|---|
| Baseline models | Dummy classifiers/regressors | Starting point was weak |
| Full model suite | 88 latest completed models | Strong model improvement over baseline |
| Neural networks | MLP classifier/regressor | Worked, but did not beat best tree/logistic models |
| Focused tuning | Tuned histogram gradient boosting | Slightly improved burnout prediction |
| Variable importance | Permutation importance | Weather signal was clearer for treatment than burnout |

**Key result from trace:**

- Latest full experiment: 88 completed models, 0 failures
- Historical experiment log: 168 rows

**Speaker note:**  
The experiment trace showed that the project improved because I kept comparing models systematically. The biggest gains came from moving beyond dummy baselines. The later gains from tuning were smaller.

## Slide 6: Final Result

**Controlled results:**

| Outcome | No Weather | Weather | Weather Change |
|---|---:|---:|---:|
| `sought_treatment` weighted F1 | 0.7962 | 0.7963 | +0.0002 |
| `burnout_index` R-squared | 0.7938 | 0.7958 | +0.0020 |

**Baseline vs best:**

| Outcome | Baseline | Best | Gain |
|---|---:|---:|---:|
| Treatment F1 | 0.3915 | 0.7963 | +0.4048 |
| Burnout R-squared | -0.0001 | 0.7958 | +0.7959 |
| Burnout RMSE | 1.9449 | 0.8788 | -1.0661 |

**Main result statement:**  
The models improved strongly over baseline, but weather only added a small amount of extra predictive value.

**Speaker note:**  
This is the most important slide. The project succeeded at prediction, but the weather effect was not large. The best evidence for weather is not the metric gain. It is the variable-importance signal in treatment prediction.

## Slide 7: Worked vs. Failed

**What worked:**

- Controlled non-weather vs weather comparisons
- Historical experiment logging
- Histogram gradient boosting for burnout prediction
- Logistic regression / gradient boosting for treatment prediction
- Variable importance for interpretation

**What partially worked:**

- Neural networks ran successfully but did not beat the best models
- Focused tuning slightly improved burnout prediction

**What did not fully work:**

- Weather did not create a large metric improvement
- Weather data was coarse
- Burnout index was mostly explained by sleep/work/health variables

**Weather importance evidence:**

| Target | Weather Feature | Importance |
|---|---|---:|
| `sought_treatment` | `wind_gust` | 0.0260 |
| `sought_treatment` | `pressure_hpa` | 0.0055 |
| `burnout_index` | `room_temperature_celsius` | 0.0032 |

**Speaker note:**  
The honest result is that weather was not a major predictor. But it was not completely useless either. The treatment model showed the clearest weather-related signal, especially wind gust.

## Slide 8: Reflection

**What I learned:**

- A strong model result is not the same as a strong weather effect.
- Baseline comparison is important because it showed the models really improved.
- Variable importance helped explain why weather had limited impact.
- Past research can still be true even if my predictive gain is small.

**Limitations:**

- Weather matching was coarse
- Exact personal weather exposure was unavailable
- Timing effects such as heat waves or sunlight lag were not modeled
- Burnout index was built from strong sleep/work indicators

**Final conclusion:**  
Weather may matter indirectly, especially for treatment prediction, but in this dataset it was not a major predictor once work, sleep, health, and mental-health variables were included.

**Speaker note:**  
My final takeaway is cautious. The project does not disprove prior weather research. It shows that in this specific dataset and prediction setup, weather had limited incremental value.

## 6-Minute Talk Timing

| Time | Slides | Focus |
|---|---|---|
| 0:00-0:40 | Slide 1-2 | Topic, claim, and research formulation |
| 0:40-1:30 | Slide 3 | Data, metrics, and baselines |
| 1:30-2:40 | Slide 4 | Agent loop and controlled design |
| 2:40-3:50 | Slide 5-6 | Experiment trace and final results |
| 3:50-4:50 | Slide 7 | What worked vs. failed |
| 4:50-6:00 | Slide 8 | Reflection, limitations, and conclusion |

## Five Required Questions

### 1. From Topic to Contract

The topic became a measurable AutoResearch contract by converting "weather affects burnout" into "test whether weather improves prediction beyond non-weather predictors."

### 2. Directions Explored

The loop explored baseline models, weather ablation, PCA burnout construction, full model suites, neural networks, focused tuning, and variable importance.

### 3. Stable Value

The stable value came from controlled comparisons, historical logging, and baseline-vs-best improvement. The models consistently beat dummy baselines.

### 4. Noise and Failure

Weather produced noisy or weak incremental gains. Neural networks worked but did not outperform the best tree/logistic models. Weather was more interpretable in treatment prediction than burnout prediction.

### 5. Limits of the Loop

The loop was limited by coarse weather matching, lack of exact exposure timing, and the fact that the burnout index was already strongly tied to sleep/work/health variables.
