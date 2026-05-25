# RandomForest Optuna optimization

We use Optuna with a seeded TPE sampler to tune RandomForest hyperparameters through training-set-only cross-validation, then refit the selected model once on the full training split before final holdout evaluation. The tuning and final refit steps are separate Kedro nodes so the best-parameter artifact is an explicit handoff, and MLflow logs consolidated trial tables instead of one child run per trial to keep the experiment view readable while preserving trial-level evidence.

## Considered Options

- Optuna TPE sampler: selected for sample-efficient optimization under a bounded local trial budget.
- Optuna grid search: rejected because exhaustive search scales poorly with fold-local preprocessing.
- Nested MLflow trial runs: rejected for now to keep the local experiment view clean.

## Consequences

- Final holdout metrics remain reporting-only and cannot influence hyperparameter selection.
- The RandomForest run now takes longer because each Optuna trial performs full cross-validation with fold-local learned preprocessing.
