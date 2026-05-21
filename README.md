# MLOps Project

Current scope: Kedro-based preprocessing and baseline modeling for the water potability dataset, including local Kaggle data download, fail-fast Great Expectations raw data validation, stratified splitting, split-wise feature engineering, training-only outlier removal, mean imputation, standard scaling, RFECV-based feature selection, final model-ready Great Expectations validation, LogisticRegression baseline training, validation/test evaluation, local MLflow tracking, and persisted reporting artifacts, with exploratory analysis still available under `notebooks/`.

## Repository Tree

```text
.
├── .gitignore                                     - Git ignore rules for virtual environments, caches, notebook checkpoints, local data, and MLflow runs.
├── README.md                                      - Project overview, current scope, preprocessing behavior, and repository conventions.
├── main.py                                        - CLI entrypoint for local data bootstrap tasks.
├── pyproject.toml                                 - Project metadata, dependencies, and Kedro project settings.
├── uv.lock                                        - Locked dependency resolution for `uv`.
├── conf/
│   ├── base/
│   │   ├── catalog.yml                            - Kedro dataset catalog for raw input, validation candidates, selected feature artifacts, model artifacts, MLflow run metadata, and reporting outputs.
│   │   └── parameters.yml                         - Runtime preprocessing, modeling, and MLflow parameters.
│   └── local/                                     - Local Kedro environment directory required by the default config loader.
├── docs/
│   └── eda_findings.md                            - Written summary of the exploratory analysis and how it motivates preprocessing choices.
├── notebooks/
│   ├── EDA.ipynb                                  - Exploratory notebook that reads the locally prepared dataset and inspects distributions, missingness, and class balance.
│   └── images/
│       └── distribution_numerical_features.png    - Exported figure of the numerical feature distributions used by the EDA report.
├── src/
│   ├── __init__.py                                - Package marker for shared source code.
│   ├── project_paths.py                           - Repo-root and data-path helpers used by notebooks and scripts.
│   └── mlops_project/
│       ├── __init__.py                            - Kedro project package marker.
│       ├── data_setup.py                          - Interactive Kaggle credential bootstrap and dataset download helper.
│       ├── datasets.py                            - Local Kedro dataset implementations for CSV, pickle, and matplotlib figure persistence.
│       ├── pipeline_registry.py                   - Registers preprocessing and modeling pipelines and composes the default Kedro pipeline.
│       ├── settings.py                            - Project settings entrypoint for Kedro.
│       └── pipelines/
│           ├── __init__.py                        - Pipeline namespace package.
│           ├── modeling/
│           │   ├── __init__.py                    - Re-exports the modeling pipeline factory.
│           │   ├── nodes.py                       - LogisticRegression training, evaluation, plotting, and MLflow logging node implementations.
│           │   └── pipeline.py                    - Kedro node graph for the baseline modeling workflow.
│           └── preprocessing/
│               ├── __init__.py                    - Re-exports the preprocessing pipeline factory.
│               ├── nodes.py                       - Split, feature-engineering, outlier-removal, imputation, scaling, and RFECV selection node implementations.
│               ├── pipeline.py                    - Kedro node graph for the preprocessing workflow.
│               └── validation.py                  - Great Expectations raw and final preprocessing contracts for the water potability dataset.
└── tests/
    ├── __init__.py                                - Test package marker.
    ├── test_data_setup.py                         - Unit tests for interactive credential bootstrap, dataset download, and CLI behavior.
    └── pipelines/
        ├── __init__.py                            - Pipeline test package marker.
        ├── modeling/
        │   ├── __init__.py                        - Modeling test package marker.
        │   ├── test_nodes.py                      - Unit tests for LogisticRegression training, evaluation, plotting, and MLflow logging behavior.
        │   └── test_pipeline.py                   - Unit tests for modeling pipeline assembly.
        └── preprocessing/
            ├── __init__.py                        - Preprocessing test package marker.
            ├── test_nodes.py                      - Unit tests for split, feature engineering, outlier detection, imputation, scaling, and RFECV behavior.
            ├── test_pipeline.py                   - Unit tests for preprocessing pipeline assembly and registry composition.
            └── test_validation.py                 - Unit tests for the fail-fast raw data contract and the final preprocessing contract.
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

## Preprocessing Behavior

- Input dataset: `data/raw/water_potability.csv`
- Target column: `Potability`
- EDA reference: `notebooks/EDA.ipynb`, with findings summarized in `docs/eda_findings.md`
- Raw data validation: Great Expectations validates the loaded dataset before splitting and raises `ValueError` on contract failure
  - Exact expected columns: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`, and `Potability`
  - Exact expected dtypes: all feature columns must be `float64`; `Potability` must be `int64`
  - Missing values are allowed only for `ph` up to `16%`, `Sulfate` up to `25%`, and `Trihalomethanes` up to `6%`
  - pH must be within `[0, 14]`; concentration-like measurements must be non-negative; `Potability` must be `0` or `1`
- Default split: stratified `70/15/15` for train/validation/test with `random_state=73`
- Feature engineering order: split first, then derive the requested ratios, interactions, stress indicators, risk scores, and binary flags on each split
- Optional validation split: set `preprocessing.validation_size` to `0` to emit an empty validation split
- Outlier detection: absolute feature Z-score threshold `> 3` on the engineered training split only with `nan_policy="omit"`
- Imputation: `SimpleImputer(strategy="mean")` fit on the cleaned training split, then applied to validation and test features
- Scaling: `StandardScaler` fit on the imputed training split, then applied to validation and test features
- Feature selection: `RFECV` with `LogisticRegression(max_iter=5000, random_state=73)`, `StratifiedKFold(n_splits=10, shuffle=True, random_state=73)`, `scoring="roc_auc"`, and `n_jobs=-1`
- Final data validation: Great Expectations validates model-ready outputs before persistence, requiring aligned split indexes, a non-empty unique selected feature list, selected feature columns in order, `float64` finite non-null features, and binary `int64` labels; empty validation outputs are allowed only when validation splitting is disabled
- Persisted outputs:
  - `X_train.pkl`, `X_validation.pkl`, `X_test.pkl`: post-RFE selected feature matrices
  - `y_train.pkl`, `y_validation.pkl`, `y_test.pkl`: split labels
  - `selected_features.pkl`: ordered training-derived selected feature list consumed by downstream holdout projection
  - `rfe_summary.csv`: per-feature RFE reporting summary with selection flag, ranking, and original feature order

## Modeling Behavior

- Baseline model: `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=73)`
- Training data: final validated `X_train` and `y_train`
- Development evaluation: final validated `X_validation` and `y_validation`
- Final holdout evaluation: final validated `X_test` and `y_test`
- Primary development metric: `validation_f1`
- Additional metrics: accuracy, precision, recall, F1, weighted F1, ROC AUC, and confusion matrix for validation and test
- MLflow tracking: local `mlruns/` with experiment `water_potability_modeling` and run `logistic_regression_baseline`
- Persisted modeling outputs:
  - `logistic_regression_model.pkl`: trained baseline model
  - `validation_metrics.csv`, `test_metrics.csv`: one-row metric tables
  - `validation_confusion_matrix.csv`, `test_confusion_matrix.csv`: 2x2 confusion matrix tables
  - `validation_confusion_matrix.png`, `test_confusion_matrix.png`: confusion matrix plots
  - `mlflow_run_info.csv`: MLflow run identifier and tracking metadata

## Running The Pipeline

1. Install dependencies with `uv sync`.
2. Run `uv run python main.py setup-data` to prepare `data/raw/water_potability.csv`.
3. Run the default Kedro pipeline with `.venv/bin/kedro run` or `uv run kedro run`.
4. Inspect persisted local outputs under `data/03_primary/`, `data/06_models/`, and `data/08_reporting/`.
5. Inspect MLflow runs with `uv run mlflow ui --backend-store-uri mlruns`.

## Data Convention

Use `src.project_paths.PROJECT_ROOT` and `src.project_paths.RAW_DATA_DIR` instead of hard-coded relative paths. This keeps notebook and script execution stable regardless of the current working directory.
