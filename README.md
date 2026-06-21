# MLOps Project

Current scope: an end-to-end Kedro MLOps proof of concept for the water potability dataset. It includes local Kaggle data setup, fail-fast Great Expectations validation, deterministic feature engineering, leakage-safe learned preprocessing, cross-validated LogisticRegression and Optuna-tuned tree-model comparison, final holdout evaluation, MLflow experiment tracking and snapshot sharing, Random Forest SHAP explainability, KS-test feature drift reporting, prediction-ready model bundles, and FastAPI serving through local or Docker execution. Exploratory analysis remains available under `notebooks/`.

## Repository Tree

```text
.
├── .gitignore                                     - Ignore rules for local environments, caches, MLflow SQLite sidecars, report exports, Kedro Viz state, and helper scripts.
├── .python-version                                - Python version selected for local tooling.
├── CONTEXT.md                                     - Project language for raw data validation, learned preprocessing, model-ready validation, model evaluation, and tuning boundaries.
├── Dockerfile                                     - Multi-stage image definition for the FastAPI prediction service.
├── MLOps_project.md                               - Original project brief, deliverables, and reference structure.
├── README.md                                      - Project overview, current scope, preprocessing behavior, and repository conventions.
├── docker-compose.yml                             - Local container configuration, model mount, port mapping, and API health check.
├── main.py                                        - CLI entrypoint for local data bootstrap and MLflow secret-audit tasks.
├── mlflow.db                                      - Committed read-only SQLite metadata snapshot migrated from existing local MLflow runs.
├── pyproject.toml                                 - Project metadata, dependencies, and Kedro project settings.
├── uv.lock                                        - Locked dependency resolution for `uv`.
├── conf/
│   ├── base/
│   │   ├── catalog.yml                            - Kedro dataset catalog for raw input, train/test artifacts, model artifacts, MLflow run metadata, and reporting outputs.
│   │   └── parameters.yml                         - Runtime preprocessing, modeling, and MLflow parameters.
│   └── local/                                     - Local Kedro environment directory required by the default config loader.
├── data/
│   ├── raw/                                       - Water potability source CSV used by preprocessing.
│   ├── 03_primary/                                - Persisted engineered train/test features and labels.
│   ├── 06_models/                                 - Persisted prediction-ready model bundles, selected features, and tuned parameters.
│   └── 08_reporting/                              - Persisted metrics, plots, SHAP summaries, drift results, and MLflow run metadata.
├── docs/
│   └── adr/
│       ├── 0001-random-forest-optuna-optimization.md - Architecture decision record for Optuna-based RandomForest tuning.
│       └── 0002-generalized-optuna-tree-model-comparison.md - Architecture decision record for shared tree-based model tuning and comparison.
├── mlruns/                                        - Committed point-in-time MLflow file-store snapshot containing run and model artifacts.
├── notebooks/
│   ├── EDA.ipynb                                  - Exploratory notebook that reads the locally prepared dataset and inspects distributions, missingness, and class balance.
│   └── images/
│       └── distribution_numerical_features.png    - Exported figure of the numerical feature distributions used by the EDA report.
├── reports/
│   ├── eda_findings.md                            - Tracked written summary of the exploratory analysis and preprocessing rationale.
│   └── report.md                                  - Final project report covering results, explainability, drift, serving, and production considerations.
├── src/
│   ├── __init__.py                                - Package marker for shared source code.
│   ├── project_paths.py                           - Repo-root and data-path helpers used by notebooks and local tooling.
│   └── mlops_project/
│       ├── __init__.py                            - Kedro project package marker.
│       ├── data_setup.py                          - Interactive Kaggle credential bootstrap and dataset download helper.
│       ├── datasets.py                            - Local Kedro dataset implementations for CSV, pickle, and matplotlib figure persistence.
│       ├── mlflow_secret_audit.py                 - Secret-like content audit helpers for local MLflow file and SQLite stores.
│       ├── modeling/
│       │   ├── __init__.py                        - Reusable modeling component package marker.
│       │   ├── evaluation.py                      - Model construction, fold-local cross-validation, final holdout evaluation, metrics, and artifact validation helpers.
│       │   ├── explainability.py                   - SHAP computation and summary plotting for the selected Random Forest bundle.
│       │   ├── experiment_tracking.py             - MLflow logging for fitted model bundles, metrics, confusion matrices, selected features, and Optuna artifacts.
│       │   ├── model_bundle.py                    - Prediction-ready persisted model bundle that applies fitted learned preprocessing before estimator prediction.
│       │   ├── optimization.py                    - Shared Optuna tuning, search-space sampling, selected-parameter resolution, and trial artifact builders for tree-based models.
│       │   ├── preprocessing.py                   - Fold-local learned preprocessing stack used during CV and final model refit.
│       │   └── validation.py                      - Great Expectations model-ready feature contract and label alignment checks after learned preprocessing.
│       ├── pipeline_registry.py                   - Registers preprocessing, aggregate modeling, per-model modeling, and default Kedro pipelines.
│       ├── settings.py                            - Project settings entrypoint for Kedro.
│       ├── pipelines/
│           ├── __init__.py                        - Pipeline namespace package.
│           ├── data_drift/
│           │   ├── __init__.py                    - Re-exports the data drift pipeline factory.
│           │   ├── nodes.py                       - Two-sample KS feature-drift calculations and report construction.
│           │   └── pipeline.py                    - Kedro node graph for comparing train and test feature distributions.
│           ├── modeling/
│           │   ├── __init__.py                    - Re-exports the modeling pipeline factory.
│           │   ├── nodes.py                       - Kedro adapters for modeling, comparison, MLflow, confusion-matrix, and SHAP workflows.
│           │   └── pipeline.py                    - Kedro node graphs for aggregate and per-model modeling workflows.
│           └── preprocessing/
│               ├── __init__.py                    - Re-exports the preprocessing pipeline factory.
│               ├── nodes.py                       - Split and deterministic feature-engineering node implementations.
│               ├── pipeline.py                    - Kedro node graph for the preprocessing workflow.
│               └── validation.py                  - Great Expectations raw-data contract and engineered train/test modeling-input checks.
│       └── serving/
│           ├── __init__.py                        - Prediction-service package marker.
│           └── app.py                             - FastAPI application exposing health and Random Forest prediction endpoints.
└── tests/
    ├── __init__.py                                - Test package marker.
    ├── test_data_setup.py                         - Unit tests for interactive credential bootstrap, dataset download, and CLI behavior.
    ├── test_mlflow_secret_audit.py                - Unit tests for MLflow secret scanning across `mlruns/` files and `mlflow.db`.
    └── pipelines/
        ├── __init__.py                            - Pipeline test package marker.
        ├── data_drift/
        │   ├── __init__.py                        - Data drift test package marker.
        │   └── test_nodes.py                      - Unit tests for KS statistics, thresholds, report shape, and drift flags.
        ├── modeling/
        │   ├── __init__.py                        - Modeling test package marker.
        │   ├── test_explainability.py             - Unit tests for SHAP output shape, ordering, values, and plots.
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

The repository currently includes a committed point-in-time snapshot of the raw dataset and generated artifacts under `data/03_primary/`, `data/06_models/`, and `data/08_reporting/`. Pipeline runs may replace these files or add new generated outputs; review those changes before committing another snapshot.

`mlruns/` and `mlflow.db` are both committed as a shareable point-in-time MLflow snapshot. The file store contains model files and run artifacts, while the SQLite database contains migrated metadata. Future local training also writes to `mlruns/`, so audit and review changes before refreshing the shared snapshot.

`reports/` stores tracked human-readable Markdown reports. `.gitignore` covers local virtual environments, Python and test caches, notebook checkpoints, MLflow SQLite sidecars, generated report exports, Kedro Viz state, and local helper scripts; it does not currently ignore `data/` or `mlruns/`.

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
- MLflow tracking: local `mlruns/` with experiment `water_potability_modeling` and separate runs for the baseline plus each tuned tree-based model; each run logs metrics, selected features, final test artifacts, and a logged pyfunc model so the MLflow UI shows it in the Models column; tuned tree-based models also log best parameters and consolidated Optuna trial tables. `mlflow.db` is a committed point-in-time read-only metadata snapshot for sharing existing runs, not the backend used for future local training runs.
- Registered pipelines: `preprocessing`, `modeling_logistic_regression`, `modeling_random_forest`, `modeling_extra_trees`, `modeling_hist_gradient_boosting`, `modeling_xgboost`, aggregate `modeling`, and `data_drift`
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

## Explainability And Drift

- The aggregate `modeling` pipeline computes SHAP values for the persisted Random Forest bundle after model comparison.
- SHAP uses the bundle's fitted learned preprocessing before explaining the estimator inputs.
- Explainability outputs:
  - `data/08_reporting/random_forest_shap_summary.csv`
  - `data/08_reporting/random_forest_shap_summary_plot.png`
- The separate `data_drift` pipeline compares each shared `X_train` and `X_test` feature with a two-sample Kolmogorov-Smirnov test.
- Features with `p_value < data_drift.significance_threshold` are marked as drifted; the default threshold is `0.05`.
- Drift output: `data/08_reporting/drift_report.csv`

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
10. Run drift detection with `uv run kedro run --pipeline data_drift`.
11. Inspect persisted outputs under `data/03_primary/`, `data/06_models/`, and `data/08_reporting/`.
12. Inspect MLflow runs with `uv run mlflow ui --backend-store-uri mlruns`.
13. Audit local MLflow stores for secret-like content with `uv run python main.py audit-mlflow-secrets`.

The per-model modeling and data drift pipelines expect preprocessing artifacts under `data/03_primary/`. Run `uv run kedro run --pipeline preprocessing` first if those artifacts are missing or stale. The aggregate `modeling` pipeline and default pipeline also produce Random Forest SHAP outputs; the standalone `modeling_random_forest` pipeline does not.

## Serving Predictions

The API loads `data/06_models/random_forest_model.pkl` at startup. Generate that artifact with the default pipeline, aggregate `modeling` pipeline, or `modeling_random_forest` pipeline before starting the service.

Run the FastAPI service locally:

```bash
uv run uvicorn mlops_project.serving.app:app --reload
```

Or build and run it with Docker Compose:

```bash
docker compose up --build
```

The service listens on `http://localhost:8000` and exposes:

- `GET /health`: confirms that the service is running and identifies the loaded model.
- `POST /predict`: accepts one JSON object containing the nine raw water-quality measurements and returns the binary prediction plus potable-class probability. `ph`, `Sulfate`, and `Trihalomethanes` may be `null`; the remaining measurements are required.
- `GET /docs`: serves the generated OpenAPI interface and request example.

## MLflow Snapshot Sharing

`mlflow.db` is a committed read-only snapshot of existing MLflow run metadata. It contains experiments, runs, params, metrics, tags, and logged-model records. It does not contain model files, plots, CSVs, or pyfunc artifacts; those remain under `mlruns/`.

The current snapshot was created from the local file store with:

```bash
uv run mlflow migrate-filestore \
  --source ./mlruns \
  --target sqlite:///./mlflow.db
```

The target database must be empty when refreshing the snapshot. If MLflow reports duplicate metric rows during migration, migrate from a temporary de-duplicated copy of `mlruns/` and leave the working `mlruns/` directory unchanged.

Before migrating or committing a refreshed snapshot, run `uv run python main.py audit-mlflow-secrets` and confirm it reports zero suspicious locations.

To inspect the snapshot on another machine, make sure both `mlflow.db` and `mlruns/` are present. The server command below is sufficient only when the artifact URIs stored inside `mlflow.db` already resolve under that machine's local `mlruns/` directory. If the copied database still points at a different checkout, regenerate the snapshot locally from `mlruns/` or rewrite the file-based `mlruns` URIs in a local copy of `mlflow.db` before starting the server.

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Open the MLflow UI URL printed by the server and inspect the `water_potability_modeling` experiment. This repository does not track a rebasing helper under `scripts/`; treat any URI-rewrite helper as local-only tooling. If the installed MLflow version requires `mlflow ui` instead of `mlflow server` for local inspection, use the same `--backend-store-uri sqlite:///mlflow.db` value.

This snapshot is not a multi-user writable backend. Continue generating new local runs through the configured `mlruns` tracking URI; refresh the snapshot only when you want to export another point-in-time view.

## Data Convention

Use `src.project_paths.PROJECT_ROOT` and `src.project_paths.RAW_DATA_DIR` instead of hard-coded relative paths. This keeps notebook and script execution stable regardless of the current working directory.
