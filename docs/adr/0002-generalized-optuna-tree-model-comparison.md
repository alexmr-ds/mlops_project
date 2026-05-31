# Generalized Optuna tree-based model comparison

We use one shared Optuna tuning and reporting pattern for tuned tree-based model families: RandomForest, ExtraTrees, HistGradientBoosting, and XGBoost. Each model performs training-set-only cross-validation through the Hyperparameter Optimization Evaluation, selects hyperparameters by the Primary Development Metric, refits once on the full training split, and reports Final Holdout Evaluation metrics without feeding them back into model selection.

## Considered Options

- Shared tuning workflow: selected because the model families need the same cross-validation boundary, artifact shape, MLflow logging behavior, and comparison report.
- Separate one-off tuning functions per model: rejected because it would duplicate the RandomForest path and make future model comparison artifacts harder to trust.
- Exhaustive Optuna grid sampling: rejected because fold-local Learned Preprocessing makes full grids expensive and the existing project decision already favors bounded seeded TPE search.

## Consequences

- Adding a new tuned tree-based model now requires estimator construction, base parameters, and a search space, rather than a new end-to-end tuning implementation.
- Per-model artifacts remain comparable because each tuned model logs the same best-parameter, CV, trial, final holdout, confusion-matrix, and MLflow metadata outputs.
- The aggregate comparison report ranks by training-set CV F1 while keeping final holdout metrics reporting-only.
