# Experiment-Result Matrix

| ID | Model | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Runtime sec | Best CV F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LogisticRegression | 0.4933 | 0.3719 | 0.3283 | 0.4933 | 0.11 | n/a |
| 2 | RandomForestClassifier | 0.4889 | 0.3702 | 0.3299 | 0.4889 | 21.11 | 0.4072 |
| 3 | PolynomialLogisticRegression | 0.4933 | 0.3697 | 0.3255 | 0.4933 | 3.17 | 0.4134 |
| 4 | GradientBoostingClassifier | 0.4933 | 0.3711 | 0.3490 | 0.4933 | 31.38 | 0.4112 |

## Result Summary

The models are effectively tied on validation accuracy. Logistic regression, polynomial logistic regression, and gradient boosting each reach 0.4933 accuracy, while random forest is slightly lower at 0.4889.

Weighted F1 is also nearly flat across the experiment set, ranging from 0.3697 to 0.3719. The best weighted F1 belongs to the baseline logistic regression, but the margin is too small to treat as a substantive modeling improvement.

The main observable difference is runtime. Logistic regression is much faster than the tree-search models, while gradient boosting is the slowest experiment.
