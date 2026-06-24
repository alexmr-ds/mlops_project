# MLOps Project

Current scope: an end-to-end Kedro MLOps proof of concept for the water potability dataset. It includes local Kaggle data setup, fail-fast Great Expectations validation, deterministic feature engineering, leakage-safe learned preprocessing, cross-validated LogisticRegression and Optuna-tuned tree-model comparison, final holdout evaluation, MLflow experiment tracking and snapshot sharing, Extra Trees SHAP explainability, KS-test feature drift reporting, prediction-ready model bundles, and FastAPI serving through local or Docker execution. Hyperparameter search is an opt-in `tuning` pipeline; the default `modeling` pipeline refits from the committed best parameters so repeated runs reproduce the same results. Exploratory analysis remains available under `notebooks/`.

## Quickstart

The fastest path from a fresh clone to seeing everything run:

```bash
uv sync
uv run python main.py setup-data              # one-time Kaggle download into data/raw/
uv run kedro run                               # preprocessing + all 5 models + SHAP
uv run kedro run --pipeline data_drift         # drift baseline + simulated production drift scenario
uv run mlflow ui --backend-store-uri mlruns    # browse the runs just created, at http://localhost:5000
```

Note that the default `kedro run` pipeline is `preprocessing + modeling` only; it does **not** include `data_drift`, which is registered as a separate pipeline and has to be run explicitly with the third command above. Skipping it means missing the simulated production drift scenario and the resulting model metric degradation, which is one of the more notable results in `reports/report.md`.

To see the Kedro pipeline graph in a browser instead of just running it:

```bash
uv sync --group viz
uv run kedro viz run
```

This opens at `http://localhost:4141`. Kedro-Viz is a separate, optional dependency group (`viz` in `pyproject.toml`) since it is a visualization tool for development, not something the model code depends on.

If you would rather inspect what is already committed than run anything yourself, see "Verifying The Report Without Running Anything" below.

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
├── pyproject.toml                                 - Project metadata, dependencies, and Kedro project settings.
├── uv.lock                                        - Locked dependency resolution for `uv`.
├── conf/
│   ├── base/
│   │   ├── catalog.yml                            - Kedro dataset catalog for raw input, train/test artifacts, model artifacts, MLflow run metadata, and reporting outputs.
│   │   └── parameters.yml                         - Runtime preprocessing, modeling, and MLflow parameters.
│   └── local/                                     - Local Kedro environment directory required by the default config loader.
├── data/
│   ├── raw/                                       - Water potability source CSV used by preprocessing.
│   ├── 03_primary/                                - Persisted train/test labels (y_train, y_test).
│   ├── 04_feature/                                - File-based feature store: engineered feature sets plus versioned metadata sidecars.
│   ├── 06_models/                                 - Persisted prediction-ready model bundles, selected features, and tuned parameters.
│   └── 08_reporting/                              - Persisted metrics, plots, SHAP summaries, drift results, and MLflow run metadata.
├── docs/
│   └── adr/
│       ├── 0001-random-forest-optuna-optimization.md - Architecture decision record for Optuna-based RandomForest tuning.
│       └── 0002-generalized-optuna-tree-model-comparison.md - Architecture decision record for shared tree-based model tuning and comparison.
├── mlflow_snapshot/                                - Frozen, read-only point-in-time MLflow export (see "MLflow Snapshot Sharing" below). Not the live tracking destination.
│   └── mlruns/                                     - Tracked file-store run and model artifacts (browsable with `mlflow ui`). The migrated `mlflow.db` is gitignored, not committed.
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
│       ├── datasets.py                            - Local Kedro dataset implementations for CSV, pickle, feature-store, and matplotlib figure persistence.
│       ├── feature_store.py                       - File-based feature store: feature definitions registry, content-addressable versioning, and offline retrieval API.
│       ├── mlflow_secret_audit.py                 - Secret-like content audit helpers for local MLflow file and SQLite stores.
│       ├── modeling/
│       │   ├── __init__.py                        - Reusable modeling component package marker.
│       │   ├── evaluation.py                      - Model construction, fold-local cross-validation, final holdout evaluation, metrics, and artifact validation helpers.
│       │   ├── explainability.py                   - SHAP computation and summary plotting for the selected champion (Extra Trees) bundle.
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
│           │   ├── nodes.py                       - Two-sample KS feature-drift calculations, simulated production drift sampling, and Extra Trees evaluation under simulated drift.
│           │   └── pipeline.py                    - Kedro node graph for the train/test drift baseline and the simulated production drift scenario.
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
│           └── app.py                             - FastAPI application exposing health and Extra Trees prediction endpoints.
└── tests/
    ├── __init__.py                                - Test package marker.
    ├── test_data_setup.py                         - Unit tests for interactive credential bootstrap, dataset download, and CLI behavior.
    ├── test_mlflow_secret_audit.py                - Unit tests for MLflow secret scanning across `mlruns/` files and `mlflow.db`.
    ├── pipelines/
    │   ├── __init__.py                            - Pipeline test package marker.
    │   ├── data_drift/
    │   │   ├── __init__.py                        - Data drift test package marker.
    │   │   └── test_nodes.py                      - Unit tests for KS statistics, thresholds, report shape, drift flags, simulated production drift sampling, and Extra Trees evaluation under simulated drift.
    │   ├── modeling/
    │   │   ├── __init__.py                        - Modeling test package marker.
    │   │   ├── test_explainability.py             - Unit tests for SHAP output shape, ordering, values, and plots.
    │   │   ├── test_nodes.py                      - Unit tests for cross-validated LogisticRegression training, tuned tree-based models, model bundles, final testing, plotting, comparison reports, and MLflow logging behavior.
    │   │   ├── test_optimization.py               - Unit tests for shared tree-based model optimization helper behavior.
    │   │   ├── test_pipeline.py                   - Unit tests for modeling pipeline assembly.
    │   │   ├── test_preprocessing.py              - Unit tests for leakage-safe model-local learned preprocessing.
    │   │   └── test_validation.py                 - Unit tests for model-ready feature and filtered-label validation contracts.
    │   └── preprocessing/
    │       ├── __init__.py                        - Preprocessing test package marker.
    │       ├── test_nodes.py                      - Unit tests for split and deterministic feature-engineering behavior.
    │       ├── test_pipeline.py                   - Unit tests for preprocessing pipeline assembly and registry composition.
    │       └── test_validation.py                 - Unit tests for fail-fast raw data and modeling input validation contracts.
    └── serving/
        ├── __init__.py                            - Serving test package marker.
        └── test_app.py                            - Unit tests for the health endpoint, prediction endpoint (full input, omitted/null nullable fields, model-not-loaded 503), and model loading.
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

The repository currently includes a committed point-in-time snapshot of the raw dataset and generated artifacts under `data/03_primary/`, `data/04_feature/`, `data/06_models/`, and `data/08_reporting/`. Pipeline runs may replace these files or add new generated outputs; review those changes before committing another snapshot.

`mlruns/` and `mlflow.db` at the repository root are **always local and gitignored**. Every machine gets its own fresh copy the first time it logs an MLflow run, because a file-store `artifact_location` is recorded as an absolute, machine-specific path and can never be safely shared between machines. The shareable, point-in-time export instead lives under the tracked `mlflow_snapshot/` directory; see "MLflow Snapshot Sharing" below.

`reports/` stores tracked human-readable Markdown reports. `.gitignore` covers local virtual environments, Python and test caches, notebook checkpoints, the local `mlruns/`/`mlflow.db` tracking state, generated report exports, Kedro Viz state, and local helper scripts; it does not currently ignore `data/`.

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
  - `data/04_feature/X_train.pkl`, `data/04_feature/X_test.pkl`: engineered feature matrices before learned preprocessing, materialised into the feature store with versioned metadata sidecars (see "Feature Store")
  - `data/03_primary/y_train.pkl`, `data/03_primary/y_test.pkl`: split labels

## Feature Store

The engineered feature matrices are materialised into a small file-based feature store under `data/04_feature/` rather than being treated as anonymous pickles. This is an own-solution feature store (no third-party service required) defined in `mlops_project.feature_store`:

- **Feature definitions registry**: `FEATURE_DEFINITIONS` is the single source of truth for every feature's name, dtype, group (`raw_measurement` or `engineered`), and human-readable description. A contract test (`tests/test_feature_store.py`) asserts the registry stays exactly in sync with the columns produced by `_engineer_feature_frame`, so the schema and the engineering code can never silently diverge.
- **Content-addressable versioning**: each materialised feature set is written by the `FeatureSetDataset` Kedro dataset together with a deterministic `<name>_metadata.json` sidecar. The metadata records a `schema_version` (hash of names plus dtypes) and a `snapshot_version` (hash of the data), combined into a single `version`. It carries no wall-clock timestamp on purpose, so re-running the pipeline on the same inputs reproduces byte-identical metadata.
- **Offline retrieval API**: `feature_store.load_offline_features("X_train")` returns the feature frame together with its metadata, the same way training, drift detection, and tests read features back from the store.
- **Scope boundary**: only the deterministic engineered features live here. The fold-local learned preprocessing (outlier filtering, imputation, scaling, RFECV) is deliberately never stored in the feature store because it must be re-fitted inside each CV fold to stay leakage-safe; it is persisted instead inside each model bundle.

Feature-store artifacts:
- `data/04_feature/{X_train,X_test,simulated_X_test}.pkl`: materialised feature sets
- `data/04_feature/{X_train,X_test,simulated_X_test}_metadata.json`: feature definitions, schema, and version for each set

## Modeling Behavior

- Baseline model: `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=73)`
- Tuned tree-based models: `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, and `XGBClassifier`, each selected by seeded Optuna TPE hyperparameter optimization with `random_state=73` fixed where supported
- Training data: engineered and validated `X_train` and `y_train`
- Development evaluation: stratified k-fold cross-validation on the training split only
- Champion model: `ExtraTreesClassifier`, which ranks first by the primary development metric (`cv_mean_f1`) in `model_comparison.csv`; it is the model used for SHAP explainability, the simulated-drift evaluation, and FastAPI serving
- Reproducibility: the default `modeling` pipeline does **not** run Optuna. It refits each tuned model from the committed `{model}_best_params.pkl` and recomputes cross-validation deterministically, so repeated runs reproduce the same artifacts. Optuna search lives in the separate, opt-in `tuning` pipeline, which is the only thing that rewrites `{model}_best_params.pkl` and the Optuna trial logs. Optuna's TPE search is sequential and floating-point sensitive, so it is not bit-reproducible across machines; decoupling it keeps the default pipeline reproducible
- Hyperparameter optimization (opt-in `tuning` pipeline): RandomForest uses `modeling.random_forest_optimization.n_trials=75`; ExtraTrees, HistGradientBoosting, and XGBoost each use `50` trials. All tuned model families maximize binary `cv_mean_f1` on training-set cross-validation folds only
- Final holdout evaluation: engineered and validated `X_test` and `y_test`, evaluated once after each model is refit on all training data
- Cross-validation config: `modeling.cross_validation.n_splits=5`, `shuffle=true`, `random_state=73`
- Fold-local learned preprocessing: outlier removal on fold-training rows only, mean imputation, standard scaling, RFECV feature selection, then model fitting
- Final learned preprocessing: the same stack is refit once on the full training split before final test evaluation
- Model-ready validation: transformed fold-training, fold-validation, final-test, and inference feature matrices are validated before estimator use
- Primary development metric: `cv_mean_f1`
- Additional metrics: accuracy, precision, recall, F1, weighted F1, ROC AUC, and confusion matrix for the final test split
- Model comparison: aggregate `modeling` writes `model_comparison.csv`, ranked by `cv_mean_f1` and including CV and final holdout accuracy, precision, recall, F1, weighted F1, and ROC AUC
- MLflow tracking: local, gitignored `mlruns/` (created fresh on first run) with experiment `water_potability_modeling` and separate runs for the baseline plus each tuned tree-based model; each run logs metrics, selected features, final test artifacts, and a logged pyfunc model so the MLflow UI shows it in the Models column; tuned tree-based models also log best parameters and consolidated Optuna trial tables. `mlflow_snapshot/` holds an archived, read-only export from an earlier machine for reference only; it is never the backend for new local training runs.
- Registered pipelines: `preprocessing`, `modeling_logistic_regression`, `modeling_random_forest`, `modeling_extra_trees`, `modeling_hist_gradient_boosting`, `modeling_xgboost`, aggregate `modeling` (all reproducible refits), opt-in `tuning` plus per-model `tuning_random_forest`, `tuning_extra_trees`, `tuning_hist_gradient_boosting`, `tuning_xgboost`, and `data_drift`
- Persisted modeling outputs:
  - `logistic_regression_model.pkl`: prediction-ready baseline bundle containing fitted learned preprocessing and the trained LogisticRegression estimator
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_model.pkl`: prediction-ready Optuna-selected bundles containing fitted learned preprocessing and the trained estimator
  - `{model_name}_selected_features.pkl`: model-specific final selected feature lists
  - `{random_forest,extra_trees,hist_gradient_boosting,xgboost}_best_params.pkl`: locked tuned hyperparameters the reproducible `modeling` pipeline refits from (rewritten only by the opt-in `tuning` pipeline)
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

- The aggregate `modeling` pipeline computes SHAP values for the persisted Extra Trees champion bundle after model comparison.
- SHAP uses the bundle's fitted learned preprocessing before explaining the estimator inputs.
- Explainability outputs:
  - `data/08_reporting/extra_trees_shap_summary.csv`
  - `data/08_reporting/extra_trees_shap_summary_plot.png`
- The separate `data_drift` pipeline runs two scenarios:
  1. A sanity baseline comparing each shared `X_train` and `X_test` feature with a two-sample Kolmogorov-Smirnov test. Since both splits come from the same random split of the same dataset, this is expected to show little to no drift.
  2. A simulated production scenario: `simulate_production_drift` perturbs the raw measurements in `X_test` with a hypothetical but realistic shift (a treatment plant changing its disinfection process: lower pH, higher chloramines, sulfate, and trihalomethanes) and recomputes the engineered features the same way training does. `evaluate_model_under_simulated_drift` then scores the persisted Extra Trees champion bundle on this sample so the resulting metric degradation is visible directly, not just inferred from the drift report.
- Features with `p_value < data_drift.significance_threshold` are marked as drifted; the default threshold is `0.05`. The simulated shift amounts are configured under `data_drift.simulated_shift` in `parameters.yml`.
- Drift outputs:
  - `data/08_reporting/drift_report.csv` (baseline X_train vs. X_test)
  - `data/08_reporting/simulated_drift_report.csv` (baseline vs. simulated production sample)
  - `data/08_reporting/simulated_drift_metrics.csv` (Extra Trees test metrics on the simulated sample, comparable to `extra_trees_test_metrics.csv`)
- The simulated drift scenario additionally requires `data/06_models/extra_trees_model.pkl` and `data/03_primary/y_test.pkl`; see "Running The Pipeline" below.

## Running The Pipeline

1. Install dependencies with `uv sync`.
2. Run `uv run python main.py setup-data` to prepare `data/raw/water_potability.csv`.
3. Run the default Kedro pipeline with `.venv/bin/kedro run` or `uv run kedro run`. The default pipeline is `preprocessing + modeling`; it refits every model from the committed best parameters (no Optuna) so it is fast and reproducible, and it does not include `data_drift` (step 10) or `tuning` (step 11).
4. Run only preprocessing with `uv run kedro run --pipeline preprocessing`.
5. Run only LogisticRegression with `uv run kedro run --pipeline modeling_logistic_regression`.
6. Run only RandomForest with `uv run kedro run --pipeline modeling_random_forest`. This refits and evaluates RandomForest from its committed best parameters (no Optuna).
7. Run only ExtraTrees (the champion) with `uv run kedro run --pipeline modeling_extra_trees`.
8. Run only HistGradientBoosting with `uv run kedro run --pipeline modeling_hist_gradient_boosting`.
9. Run only XGBoost with `uv run kedro run --pipeline modeling_xgboost`.
10. Run drift detection with `uv run kedro run --pipeline data_drift`.
11. (Optional, not reproducible) Re-run Optuna hyperparameter search with `uv run kedro run --pipeline tuning` (or per model, e.g. `--pipeline tuning_extra_trees`). This rewrites the committed `{model}_best_params.pkl` and Optuna trial logs; only run it when you deliberately want to re-tune, then re-run `modeling` to refresh the artifacts.
12. Inspect persisted outputs under `data/03_primary/`, `data/04_feature/`, `data/06_models/`, and `data/08_reporting/`.
13. Inspect MLflow runs with `uv run mlflow ui --backend-store-uri mlruns` (only after running at least one modeling pipeline: `mlruns/` is gitignored and does not exist until the first local run creates it; the UI starts fine on a fresh clone but shows no experiments until then). To browse historical runs without training anything yourself, see the read-only `mlflow_snapshot/` export under "MLflow Snapshot Sharing" below instead.
14. Audit local MLflow stores for secret-like content with `uv run python main.py audit-mlflow-secrets`.
15. Visualize the pipeline graph with `uv sync --group viz` then `uv run kedro viz run` (serves at `http://localhost:4141`). This is the optional `viz` dependency group in `pyproject.toml`, separate from the model code.

The per-model modeling and data drift pipelines expect preprocessing artifacts: the engineered feature sets under `data/04_feature/` and the labels under `data/03_primary/`. Run `uv run kedro run --pipeline preprocessing` first if those artifacts are missing or stale. The aggregate `modeling` pipeline and default pipeline also produce Extra Trees SHAP outputs; the standalone `modeling_extra_trees` pipeline does not.

The `data_drift` pipeline additionally needs `data/06_models/extra_trees_model.pkl` and `data/03_primary/y_test.pkl` for its simulated-drift evaluation step (`evaluate_model_under_simulated_drift_node`), which scores the trained Extra Trees champion on a simulated production sample. Run `uv run kedro run --pipeline modeling_extra_trees` (or `modeling`) first if the model artifact is missing.

## Serving Predictions

The API loads `data/06_models/extra_trees_model.pkl` at startup. Generate that artifact with the default pipeline, aggregate `modeling` pipeline, or `modeling_extra_trees` pipeline before starting the service.

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

`mlruns/` and `mlflow.db` at the repository root are **never committed**: they are gitignored and re-created fresh the first time anyone runs the modeling pipeline on their own machine. This is a hard requirement, not a style preference: a local file-store records `artifact_location` as an absolute, machine-specific path (e.g. `/Users/alexandre/Documents/mlops_project/mlruns/...`), so committing the live tracking directory and reusing it on another machine causes every artifact-logging call to fail with `PermissionError`.

Instead, `mlflow_snapshot/mlruns/` is a **tracked, read-only, point-in-time file-store export**. It exists purely so a grader or teammate can browse historical params, metrics, and tags without re-running the pipeline. Because of the same absolute-path limitation, the artifact URIs recorded inside this snapshot point at the machine that generated it: metrics, params, and tags browse fine, but resolving model files, plots, or CSVs through the MLflow UI will not work on a different machine. Treat the snapshot as a reference for numbers, not a working backend.

Browse the archived snapshot directly from the committed file store:

```bash
uv run mlflow ui --backend-store-uri ./mlflow_snapshot/mlruns
```

Open the MLflow UI URL printed and inspect the `water_potability_modeling` experiment.

The migrated SQLite form (`mlflow_snapshot/mlflow.db`) is **not committed**; it is gitignored. GitHub's push protection flags it as a false-positive "Lob Test API Key" because the binary packs the metric key `test_*` immediately next to a 32-character run id (e.g. `test_accuracy<run_id>`). The file store above carries the same metrics and params and has no such false positive, so it is what we ship. If you specifically want the SQLite metadata DB locally, regenerate it from the tracked file store:

```bash
uv run mlflow migrate-filestore \
  --source ./mlflow_snapshot/mlruns \
  --target sqlite:///./mlflow_snapshot/mlflow.db
```

If MLflow reports duplicate metric rows during migration, migrate from a temporary de-duplicated copy of the file store (this project's MLflow version writes each metric history row twice). Before refreshing the committed file-store snapshot from a new training run, run `uv run python main.py audit-mlflow-secrets --tracking-dir mlflow_snapshot/mlruns` and confirm it reports zero suspicious locations.

This snapshot is not a multi-user writable backend. New local runs always go through the local, gitignored `mlruns` tracking URI; refresh the snapshot only when you want to export another point-in-time view for sharing.

## Verifying The Report Without Running Anything

Every number, table, and plot referenced in `reports/report.md` is backed by a file already committed to this repository, so a grader can check them directly without installing anything or running the pipeline:

- Model comparison table (Section 3.3): `data/08_reporting/model_comparison.csv`, and per-model detail in `data/08_reporting/{model_name}_test_metrics.csv` and `{model_name}_test_confusion_matrix.{csv,png}`.
- SHAP feature importance (Section 3.4): `data/08_reporting/extra_trees_shap_summary.csv` and `extra_trees_shap_summary_plot.png`.
- Drift baseline and simulated production scenario (Section 4.2): `data/08_reporting/drift_report.csv`, `simulated_drift_report.csv`, and `simulated_drift_metrics.csv` (compare against `extra_trees_test_metrics.csv` for the before/after degradation).
- Trained model bundles themselves: `data/06_models/{model_name}_model.pkl`.
- MLflow run history (params, metrics, tags, per-run artifacts) without training anything: the committed `mlflow_snapshot/mlruns/` file store, browsable with `mlflow ui --backend-store-uri ./mlflow_snapshot/mlruns` (see "MLflow Snapshot Sharing" above). Note that the live `mlflow.db` and `mlruns/` at the repository root are **not** committed (gitignored, local-only, recreated on first run), and the migrated `mlflow_snapshot/mlflow.db` is also gitignored (regenerate it locally from the file store if needed).

## Data Convention

Use `src.project_paths.PROJECT_ROOT` and `src.project_paths.RAW_DATA_DIR` instead of hard-coded relative paths. This keeps notebook and script execution stable regardless of the current working directory.
