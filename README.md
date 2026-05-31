# MLOps Project

Current scope: Kedro-based preprocessing and registered per-model modeling for the water potability dataset, including local Kaggle data download, fail-fast Great Expectations raw data validation, stratified train/test splitting, split-wise deterministic feature engineering, fold-local learned preprocessing, fold-local model-ready feature validation, k-fold cross-validation, LogisticRegression baseline training, Optuna-optimized tree-based model comparison across RandomForest, ExtraTrees, HistGradientBoosting, and XGBoost, final holdout test evaluation, local MLflow tracking with logged pyfunc models, prediction-ready persisted model bundles, and persisted reporting artifacts, with exploratory analysis still available under `notebooks/`.

## Repository Tree

```text
.
├── .gitignore                                     - Git ignore rules for virtual environments, caches, notebook checkpoints, local data, and MLflow runs.
├── CONTEXT.md                                     - Project language for raw data validation, learned preprocessing, model-ready validation, model evaluation, and tuning boundaries.
├── README.md                                      - Project overview, current scope, preprocessing behavior, and repository conventions.
├── main.py                                        - CLI entrypoint for local data bootstrap tasks.
├── pyproject.toml                                 - Project metadata, dependencies, and Kedro project settings.
├── uv.lock                                        - Locked dependency resolution for `uv`.
├── conf/
│   ├── base/
│   │   ├── catalog.yml                            - Kedro dataset catalog for raw input, train/test artifacts, model artifacts, MLflow run metadata, and reporting outputs.
│   │   └── parameters.yml                         - Runtime preprocessing, modeling, and MLflow parameters.
│   └── local/                                     - Local Kedro environment directory required by the default config loader.
├── docs/
│   └── adr/
│       ├── 0001-random-forest-optuna-optimization.md - Architecture decision record for Optuna-based RandomForest tuning.
│       └── 0002-generalized-optuna-tree-model-comparison.md - Architecture decision record for shared tree-based model tuning and comparison.
├── notebooks/
│   ├── EDA.ipynb                                  - Exploratory notebook that reads the locally prepared dataset and inspects distributions, missingness, and class balance.
│   └── images/
│       └── distribution_numerical_features.png    - Exported figure of the numerical feature distributions used by the EDA report.
├── reports/
│   └── eda_findings.md                            - Tracked written summary of the exploratory analysis and preprocessing rationale.
├── src/
│   ├── __init__.py                                - Package marker for shared source code.
│   ├── project_paths.py                           - Repo-root and data-path helpers used by notebooks and scripts.
│   └── mlops_project/
│       ├── __init__.py                            - Kedro project package marker.
│       ├── data_setup.py                          - Interactive Kaggle credential bootstrap and dataset download helper.
│       ├── datasets.py                            - Local Kedro dataset implementations for CSV, pickle, and matplotlib figure persistence.
│       ├── modeling/
│       │   ├── __init__.py                        - Reusable modeling component package marker.
│       │   ├── evaluation.py                      - Model construction, fold-local cross-validation, final holdout evaluation, metrics, and artifact validation helpers.
│       │   ├── experiment_tracking.py             - MLflow logging for fitted model bundles, metrics, confusion matrices, selected features, and Optuna artifacts.
│       │   ├── model_bundle.py                    - Prediction-ready persisted model bundle that applies fitted learned preprocessing before estimator prediction.
│       │   ├── optimization.py                    - Shared Optuna tuning, search-space sampling, selected-parameter resolution, and trial artifact builders for tree-based models.
│       │   ├── preprocessing.py                   - Fold-local learned preprocessing stack used during CV and final model refit.
│       │   └── validation.py                      - Great Expectations model-ready feature contract and label alignment checks after learned preprocessing.
│       ├── pipeline_registry.py                   - Registers preprocessing, aggregate modeling, per-model modeling, and default Kedro pipelines.
│       ├── settings.py                            - Project settings entrypoint for Kedro.
│       └── pipelines/
│           ├── __init__.py                        - Pipeline namespace package.
│           ├── modeling/
│           │   ├── __init__.py                    - Re-exports the modeling pipeline factory.
│           │   ├── nodes.py                       - Kedro adapters for modeling workflows and final confusion-matrix plot creation.
│           │   └── pipeline.py                    - Kedro node graphs for aggregate and per-model modeling workflows.
│           └── preprocessing/
│               ├── __init__.py                    - Re-exports the preprocessing pipeline factory.
│               ├── nodes.py                       - Split and deterministic feature-engineering node implementations.
│               ├── pipeline.py                    - Kedro node graph for the preprocessing workflow.
│               └── validation.py                  - Great Expectations raw-data contract and engineered train/test modeling-input checks.
└── tests/
    ├── __init__.py                                - Test package marker.
    ├── test_data_setup.py                         - Unit tests for interactive credential bootstrap, dataset download, and CLI behavior.
    └── pipelines/
        ├── __init__.py                            - Pipeline test package marker.
        ├── modeling/
        │   ├── __init__.py                        - Modeling test package marker.
        │   ├── test_nodes.py                      - Unit tests for cross-validated LogisticRegression training, tuned tree-based models, model bundles, final testing, plotting, comparison reports, and MLflow logging behavior.
        │   ├── test_optimization.py               - Unit tests for shared tree-based model optimization helper behavior.
        │   ├── test_pipeline.py                   - Unit tests for modeling pipeline assembly.
        │   ├── test_preprocessing.py              - Unit tests for leakage-safe model-local learned preprocessing.
        │   └── test_validation.py                 - Unit tests for model-ready feature and filtered-label validation contracts.
        └── preprocessing/
            ├── __init__.py                        - Preprocessing test package marker.
            ├── test_nodes.py                      - Unit tests for split and deterministic feature-engineering behavior.
            ├── test_pipeline.py                   - Unit tests for preprocessing pipeline assembly and registry composition.
            └── test_validation.py                 - Unit tests for fail-fast raw data and modeling input validation contracts.
```

## Local Data Setup

1. Install dependencies with `uv sync`.
2. Run `uv run python main.py setup-data`.
3. If `~/.kaggle/kaggle.json` is missing, the script will:
   - explain that Kaggle API credentials are required
   - link you to `https://www.kaggle.com/settings/api`
   - ask whether you want it to create `~/.kaggle/kaggle.json`
   - prompt for `username` and `key`
   - set file permissions to `600`
4. The script then downloads `adityakadiwal/water-potability` into local `data/raw/`.

`data/` is intentionally local-only and ignored by Git. That includes the downloaded raw CSV, generated pipeline outputs under `data/03_primary/`, and reporting artifacts under `data/08_reporting/`.

`mlruns/` is also local-only and ignored by Git. It stores MLflow runs for local development.

`reports/` stores tracked human-readable Markdown reports. Generated local pipeline reporting artifacts remain under ignored `data/08_reporting/`.

## Preprocessing Behavior

- Input dataset: `data/raw/water_potability.csv`
- Target column: `Potability`
- EDA reference: `notebooks/EDA.ipynb`, with findings summarized in `reports/eda_findings.md`
- Raw data validation: Great Expectations validates the loaded dataset before splitting and raises `ValueError` on contract failure
  - Exact expected columns: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`, and `Potability`
  - Exact expected dtypes: all feature columns must be `float64`; `Potability` must be `int64`
  - Missing values are allowed only for `ph` up to `16%`, `Sulfate` up to `25%`, and `Trihalomethanes` up to `6%`
  - pH must be within `[0, 14]`; concentration-like measurements must be non-negative; `Potability` must be `0` or `1`
- Default split: stratified `85/15` for train/final-test with `random_state=73`
- Feature engineering order: split first, then derive the requested ratios, interactions, stress indicators, risk scores, and binary flags on each split
- Modeling-input validation: validates aligned train/test indexes, matching feature-column order, numeric engineered features, finite-or-missing feature values, and binary `int64` labels before persistence
- Learned preprocessing is intentionally no longer fit in the preprocessing pipeline. `mlops_project.modeling.preprocessing.ModelPreprocessor` fits outlier filtering, imputation, scaling, and RFECV feature selection inside each cross-validation fold and inside the final full-training refit to avoid leakage. The final fitted preprocessing stack is persisted with each model bundle.
- Model-ready feature validation: Great Expectations validates transformed estimator inputs after learned preprocessing for non-empty rows, exact selected feature columns and order, numeric values, and no nulls; pandas/numpy checks reject infinite feature values and invalid filtered-label alignment before estimator fitting or prediction.
- Persisted outputs:
  - `X_train.pkl`, `X_test.pkl`: engineered feature matrices before learned preprocessing
  - `y_train.pkl`, `y_test.pkl`: split labels

## Modeling Behavior

- Baseline model: `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=73)`
- Tuned tree-based models: `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, and `XGBClassifier`, each selected by seeded Optuna TPE hyperparameter optimization with `random_state=73` fixed where supported
- Training data: engineered and validated `X_train` and `y_train`
- Development evaluation: stratified k-fold cross-validation on the training split only
- Hyperparameter optimization: RandomForest uses `modeling.random_forest_optimization.n_trials=75` by default; ExtraTrees, HistGradientBoosting, and XGBoost each use `50` trials by default. All tuned model families maximize binary `cv_mean_f1` on training-set cross-validation folds only
- Final holdout evaluation: engineered and validated `X_test` and `y_test`, evaluated once after each model is refit on all training data
- Cross-validation config: `modeling.cross_validation.n_splits=5`, `shuffle=true`, `random_state=73`
- Fold-local learned preprocessing: outlier removal on fold-training rows only, mean imputation, standard scaling, RFECV feature selection, then model fitting
- Final learned preprocessing: the same stack is refit once on the full training split before final test evaluation
- Model-ready validation: transformed fold-training, fold-validation, final-test, and inference feature matrices are validated before estimator use
- Primary development metric: `cv_mean_f1`
- Additional metrics: accuracy, precision, recall, F1, weighted F1, ROC AUC, and confusion matrix for the final test split
- Model comparison: aggregate `modeling` writes `model_comparison.csv`, ranked by `cv_mean_f1` and including CV and final holdout accuracy, precision, recall, F1, weighted F1, and ROC AUC
- MLflow tracking: local `mlruns/` with experiment `water_potability_modeling` and separate runs for the baseline plus each tuned tree-based model; each run logs metrics, selected features, final test artifacts, and a logged pyfunc model so the MLflow UI shows it in the Models column; tuned tree-based models also log best parameters and consolidated Optuna trial tables
- Registered model pipelines: `modeling_logistic_regression`, `modeling_random_forest`, `modeling_extra_trees`, `modeling_hist_gradient_boosting`, `modeling_xgboost`, and aggregate `modeling`
- Persisted modeling outputs:
  - `logistic_regression_model.pkl`: prediction-ready baseline bundle containing fitted learned preprocessing and the trained LogisticRegression estimator
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_model.pkl`: prediction-ready Optuna-selected bundles containing fitted learned preprocessing and the trained estimator
  - `{model_name}_selected_features.pkl`: model-specific final selected feature lists
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_best_params.pkl`: selected tuned hyperparameters used for final refit
  - `{model_name}_cv_metrics.csv`: one-row CV summary metric tables
  - `{model_name}_cv_fold_metrics.csv`: per-fold metric tables
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_optuna_trials.csv`: one-row-per-trial Optuna summary tables with sampled parameters and CV metrics
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_optuna_fold_metrics.csv`: one-row-per-trial-fold Optuna fold metrics tables
  - `{model_name}_test_metrics.csv`: one-row final test metric tables
  - `{model_name}_test_confusion_matrix.csv`: 2x2 final test confusion matrix tables
  - `{model_name}_test_confusion_matrix.png`: final test confusion matrix plots
  - `model_comparison.csv`: aggregate model comparison ranked by the Primary Development Metric
  - `mlflow_run_info.csv`: MLflow run identifier and tracking metadata
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_mlflow_run_info.csv`: tuned model MLflow run identifiers and tracking metadata

## Running The Pipeline

1. Install dependencies with `uv sync`.
2. Run `uv run python main.py setup-data` to prepare `data/raw/water_potability.csv`.
3. Run the default Kedro pipeline with `.venv/bin/kedro run` or `uv run kedro run`.
4. Run only preprocessing with `uv run kedro run --pipeline preprocessing`.
5. Run only LogisticRegression with `uv run kedro run --pipeline modeling_logistic_regression`.
6. Run only RandomForest with `uv run kedro run --pipeline modeling_random_forest`. This runs Optuna tuning first, then refits and evaluates the selected RandomForest once on the final holdout split.
7. Run only ExtraTrees with `uv run kedro run --pipeline modeling_extra_trees`.
8. Run only HistGradientBoosting with `uv run kedro run --pipeline modeling_hist_gradient_boosting`.
9. Run only XGBoost with `uv run kedro run --pipeline modeling_xgboost`.
10. Inspect persisted local outputs under `data/03_primary/`, `data/06_models/`, and `data/08_reporting/`.
11. Inspect MLflow runs with `uv run mlflow ui --backend-store-uri mlruns`.

The per-model modeling pipelines expect the final preprocessing artifacts under `data/03_primary/`. Run `uv run kedro run --pipeline preprocessing` first if those artifacts are missing or stale.

## Data Convention

Use `src.project_paths.PROJECT_ROOT` and `src.project_paths.RAW_DATA_DIR` instead of hard-coded relative paths. This keeps notebook and script execution stable regardless of the current working directory.
