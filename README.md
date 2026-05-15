# MLOps Project

Current scope: Kedro-based preprocessing for the water potability dataset, including local Kaggle data download, stratified splitting, training-only outlier removal, mean imputation, and standard scaling, with exploratory analysis still available under `notebooks/`.

## Repository Tree

```text
.
├── .gitignore                                     - Git ignore rules for virtual environments, caches, notebook checkpoints, and local data.
├── README.md                                      - Project overview, current scope, and repository conventions.
├── main.py                                        - Minimal CLI entrypoint placeholder.
├── pyproject.toml                                 - Project metadata, dependencies, and Kedro project settings.
├── uv.lock                                        - Locked dependency resolution for `uv`.
├── conf/
│   ├── base/
│   │   ├── catalog.yml                            - Kedro dataset catalog for raw input, intermediates, and persisted preprocessing outputs.
│   │   └── parameters.yml                         - Runtime preprocessing parameters such as split ratios and outlier threshold.
│   └── local/                                     - Local Kedro environment directory required by the default config loader.
├── notebooks/
│   └── EDA.ipynb                                  - Exploratory notebook that authenticates with Kaggle, downloads the dataset locally, and inspects distributions and missing values.
├── src/
│   ├── __init__.py                                - Package marker for shared source code.
│   ├── project_paths.py                           - Repo-root and data-path helpers used by notebooks and scripts.
│   └── mlops_project/
│       ├── __init__.py                            - Kedro project package marker.
│       ├── datasets.py                            - Local Kedro dataset implementations for CSV and pickle persistence.
│       ├── pipeline_registry.py                   - Registers the preprocessing pipeline and sets it as the default Kedro pipeline.
│       ├── settings.py                            - Project settings entrypoint for Kedro.
│       └── pipelines/
│           ├── __init__.py                        - Pipeline namespace package.
│           └── preprocessing/
│               ├── __init__.py                    - Re-exports the preprocessing pipeline factory.
│               ├── nodes.py                       - Split, outlier-removal, mean-imputation, and scaling node implementations.
│               └── pipeline.py                    - Kedro node graph for the preprocessing workflow.
└── tests/
    ├── __init__.py                                - Test package marker.
    └── pipelines/
        ├── __init__.py                            - Pipeline test package marker.
        └── preprocessing/
            ├── __init__.py                        - Preprocessing test package marker.
            ├── test_nodes.py                      - Unit tests for split, outlier detection, and scaling behavior.
            └── test_pipeline.py                   - Unit tests for pipeline assembly and registration.
```

## Local Data Setup

1. Install dependencies with `uv sync`.
2. Create the Kaggle credentials directory with `mkdir -p ~/.kaggle`.
3. Create `~/.kaggle/kaggle.json` with this shape:

   ```json
   {
     "username": "<your-kaggle-username>",
     "key": "<your-kaggle-api-key>"
   }
   ```

4. Generate the API key from `https://www.kaggle.com/settings/api`.
5. Restrict file permissions with `chmod 600 ~/.kaggle/kaggle.json`.
6. Run the download cell in `notebooks/EDA.ipynb` to fetch `water_potability.csv` into local `data/raw/`.

`data/` is intentionally local-only and ignored by Git. That includes the downloaded raw CSV and generated Kedro outputs under `data/03_primary/`.

## Preprocessing Behavior

- Input dataset: `data/raw/water_potability.csv`
- Target column: `Potability`
- Default split: stratified `70/15/15` for train/validation/test
- Optional validation: set `preprocessing.validation_size` to `0` to emit an empty validation split
- Outlier detection: absolute feature Z-score threshold `> 3` on the training split only with `nan_policy="omit"`
- Imputation: `SimpleImputer(strategy="mean")` fit on the cleaned training split, then applied to validation and test features
- Scaling: `StandardScaler` fit on the imputed training split, then applied to validation and test features

## Running The Pipeline

1. Install dependencies with `uv sync`.
2. Ensure `data/raw/water_potability.csv` has been downloaded locally from `notebooks/EDA.ipynb`.
3. Run the default Kedro pipeline with `.venv/bin/kedro run` or `uv run kedro run`.
4. Inspect persisted local outputs under `data/03_primary/`.

## Data Convention

Use `src.project_paths.PROJECT_ROOT` and `src.project_paths.RAW_DATA_DIR` instead of hard-coded relative paths. This keeps notebook and script execution stable regardless of the current working directory.
