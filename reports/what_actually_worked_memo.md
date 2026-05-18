# What Actually Worked Memo

## What Worked

The reproducible experiment setup worked. The project now has a locked train/validation/test split, saved metrics, saved diagnostics, and a clear comparison matrix. This is valuable because it prevents accidental test-set use and makes every model change comparable.

The simple logistic regression baseline also worked as a strong reference point. It was not highly accurate, but it matched or beat every completed alternative while running far faster.

## What Did Not Work

Adding generic model complexity did not work. Random forest, polynomial logistic regression, and gradient boosting did not improve validation accuracy. They also did not fix the main class-level failure.

The main model behavior is still collapse toward rating `3`. This means the model can look acceptable on accuracy while failing to identify ratings `1`, `4`, and `5`.

## What Was Inconclusive

Seasonality and state temperature variables are inconclusive. The code changes were started, but the expanded run was stopped before valid metrics were written. These features should be tested in a smaller controlled run before making a claim.

## Practical Takeaway

The next useful experiment is not another broad model search. The next useful experiment should directly attack the failure mode:

1. Use macro F1 or balanced accuracy as the search metric.
2. Add sample weights or class weights.
3. Track per-class recall, especially classes `1` and `5`.
4. Run the new seasonality and temperature feature set with a small grid first, then expand only if it improves validation performance.

## Current Best Answer

The current best completed model is still logistic regression. It is the model to keep until a future completed experiment beats it on validation accuracy or weighted F1 without damaging minority-class recall.
