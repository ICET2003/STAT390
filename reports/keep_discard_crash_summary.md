# Keep / Discard / Crash Summary

| Experiment | Status | Decision | Reason |
|---|---|---|---|
| LogisticRegression | Complete | Keep | Fastest run, best weighted F1, tied best validation accuracy |
| RandomForestClassifier | Complete | Discard | Lower validation accuracy than baseline and much slower |
| PolynomialLogisticRegression | Complete | Discard | Ties accuracy but has lower weighted F1 than baseline |
| GradientBoostingClassifier | Complete | Discard for now | Ties accuracy but is much slower and does not beat baseline F1 |
| Expanded tuning plus seasonality/state temperature | Stopped | Crash/stopped | Interrupted before a valid result file was produced |

## Keep

Keep `LogisticRegression` as the current baseline and comparison point. It reaches 0.4933 validation accuracy and 0.3719 weighted F1 in 0.11 seconds.

## Discard

Discard the completed tree and polynomial variants as final candidates for now. They do not create a meaningful validation gain, and the tree models cost substantially more runtime.

## Crash / Stopped

The expanded run with seasonality, temperature proxies, and larger grids was stopped before completion. It should not be cited as evidence for or against the feature change until it completes and writes metrics.
