# Final Two-Week Plan

## Week 1: Lock the Research Design

### Day 1

- Finalize the research question and target definitions.
- Use `burnout_index` as the main target.
- Keep `sought_treatment` as a secondary mental-health comparison target.

### Day 2

- Use the historical weather source for the main result.
- Keep OpenWeatherMap current weather as a robustness/reference comparison only.
- Confirm the weather variables included in the final model.

### Day 3

- Run the full experiment suite:
  - treatment without weather
  - treatment with weather
  - burnout index without weather
  - burnout index with weather
- Confirm `results/historical_experiment_log.csv` is updated.

### Day 4

- Add robustness checks:
  - compare weather vs climate-region-only features
  - compare variable importance for weather and non-weather models
  - rerun with one alternate random seed if time allows

### Day 5

- Freeze the best result tables.
- Export final plots.
- Write the results interpretation.

## Week 2: Final Report and Presentation

### Day 6

- Draft methods section:
  - data sources
  - PCA burnout index
  - weather merge
  - model families
  - validation design

### Day 7

- Draft results section:
  - ablation table
  - best model comparison
  - weather vs non-weather interpretation
  - variable-importance interpretation

### Day 8

- Draft limitations section:
  - state-level weather is coarse
  - weather effects may be indirect and absorbed by sleep, work, and health variables
  - treatment is mental-health related but not identical to burnout
  - predictive improvement is small

### Day 9

- Build final presentation slides.
- Include:
  - project motivation
  - pipeline diagram
  - PCA index design
  - ablation table
  - final conclusion

### Day 10

- Final cleanup:
  - verify existing `python scripts/run_full_experiments.py` outputs unless results changed
  - verify report files
  - proofread README and project statement
  - prepare final submission

## Final Deliverables

- Revised project statement: `docs/program.md`
- Ablation table: `reports/ablation_comparison_table.md`
- Full result matrix: `reports/experiment_result_matrix.csv`
- Historical log: `results/historical_experiment_log.csv`
- Final report bundle: `reports/complete_experiment_log_bundle.md`
- Final two-week plan: `reports/final_two_week_plan.md`
- Variable importance report: `reports/variable_importance.md`
