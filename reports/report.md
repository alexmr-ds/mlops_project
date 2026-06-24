# MLOps Project Report: Water Potability Classification

---

## 1. Motivation and Success Metrics

Access to clean drinking water is one of the most fundamental public health challenges worldwide. According to the WHO, approximately 2 billion people lack access to safe water at home, making automated water quality assessment a practically meaningful problem. We chose the Kaggle Water Potability dataset because it represents a realistic scenario where automated screening could supplement traditional laboratory testing, particularly in resource-constrained settings where testing every sample manually is not feasible.

The dataset contains 3,276 water samples, each described by nine physicochemical measurements: pH, hardness, total dissolved solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The binary target indicates whether a sample is safe to drink (1) or not (0).

### Success Metrics

We defined success along two dimensions before any modelling began:

- **Primary metric (ROC-AUC ≥ 0.65 on the test split):** ROC-AUC is threshold-independent and handles class imbalance better than accuracy, making it the right choice for comparing models across the development phase.
- **Secondary metric (test F1 ≥ 0.45 for the positive class):** In a water safety context, failing to flag unsafe water (false negatives) is more costly than wrongly flagging safe water, so recall matters alongside precision.

We considered our pipeline successful if the best model cleared both thresholds on the untouched test split.

---

## 2. Project Planning

We organised the work into four iterative sprints loosely inspired by the agile methodology. Each sprint had a clear deliverable and a definition of done.

### Sprint 1: Exploratory Analysis and Data Contract (Week 1)

We started by exploring the dataset in `notebooks/EDA.ipynb` to understand feature distributions, missingness, and class balance before writing a single line of pipeline code. The main findings (near-Gaussian distributions, three nullable features, and a 61/39 class split) directly shaped every preprocessing decision made in Sprint 2. We also defined a Great Expectations raw data contract as the formal definition of what constitutes acceptable input, encoding the agreed missingness limits and physically plausible ranges for each measurement.

**Deliverable:** EDA notebook, `reports/eda_findings.md`, and a fail-fast Great Expectations validation node in the Kedro pipeline.

### Sprint 2: Preprocessing Pipeline (Week 2)

With the data contract in place, we built the full Kedro preprocessing pipeline: stratified 85/15 train/test splitting, deterministic feature engineering (23 derived features from domain knowledge), training-only outlier removal, fold-local mean imputation and standard scaling, and RFECV-based feature selection. The key architectural decision was to move all learned preprocessing inside each CV fold rather than fitting it once on the full training set, which eliminates any leakage of validation statistics into the feature transformation.

**Deliverable:** `src/mlops_project/pipelines/preprocessing/`, 26 unit tests, and persisted `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`.

### Sprint 3: Modelling, Tuning, and MLflow (Weeks 3-4)

We trained five model families in order of complexity: a logistic regression baseline followed by four tree-based classifiers (Random Forest, Extra Trees, Histogram Gradient Boosting, XGBoost). For each tree model, we ran an Optuna hyperparameter search on training-set-only stratified CV before fitting the final model on the full training set and evaluating it once on the test split. All experiments were tracked in MLflow, with per-run artifacts including CV fold metrics, Optuna trial logs, the best hyperparameters, and the serialised model bundle.

**Deliverable:** `src/mlops_project/pipelines/modeling/`, MLflow experiment `water_potability_modeling`, `data/08_reporting/model_comparison.csv`.

### Sprint 4: Explainability, Drift Detection, and Serving (Week 5)

In the final sprint we added the three production-readiness components. SHAP explainability was added for the best-performing model (Extra Trees) to understand which features drive predictions. A data drift detection pipeline using KS tests was built to monitor whether the feature distributions seen in deployment diverge from the training baseline. Finally, we containerised the Extra Trees model bundle as a FastAPI REST API so the classifier can be queried without any Python environment setup.

**Deliverable:** SHAP summary plot, `data/08_reporting/drift_report.csv` and `simulated_drift_report.csv`, `Dockerfile`, `docker-compose.yml`, and this report.

### Pipeline Structure

The brief's reference layout splits work into seven granular pipelines (`data_quality`, `data_cleaning`, `data_feat_engineering`, `data_split`, `model_train`, `model_selection`, `model_predict`), described as a preference rather than a requirement. We consolidated into `preprocessing` (validation through feature engineering and splitting) and `modeling` (training, evaluation, and MLflow logging for all five model families), plus a separate `data_drift` pipeline. The reason is that our preprocessing steps share a single fitted state per CV fold (imputer, scaler, RFECV selector) that has to move through validation, cleaning, and feature engineering as one unit to stay leakage-safe; splitting it into separate pipelines would mean re-loading and re-serialising that state between stages for no benefit. Crucially, hyperparameter search is split into its own opt-in `tuning` pipeline: Optuna's TPE sampler is sequential and floating-point sensitive, so re-tuning is not bit-reproducible across machines. The default `modeling` pipeline therefore never tunes; it refits each model from the committed `best_params`, so repeated runs reproduce identical artifacts, while `tuning` is the only thing that rewrites those parameters. Each pipeline can still be run independently (`kedro run --pipeline data_drift`), which is the property the brief's structure is actually aiming for.

---

## 3. Data Exploration and Modelling Results

### 3.1 Exploratory Analysis

The EDA revealed three important characteristics of the dataset, visible in the feature distributions below:

![Distribution of numerical features](../notebooks/images/distribution_numerical_features.png)

**Near-Gaussian feature distributions.** All nine measurements have skewness below 0.7 in absolute value, which justifies parametric preprocessing: Z-score outlier removal, mean imputation, and standard scaling all assume roughly symmetric distributions.

**Selective missingness.** Only pH (15.0 %), sulfate (23.8 %), and trihalomethanes (4.9 %) have missing values. All three are nullable by domain convention: instruments occasionally produce out-of-range readings that are recorded as absent rather than zero. Mean imputation fitted per training fold handles these gaps without leaking evaluation-split statistics into the transform.

**Moderate class imbalance.** 39 % of samples are potable, 61 % are not. Stratified splitting and CV folds preserve this ratio across every subset, and ROC-AUC is the primary development metric precisely because it is threshold-invariant.

### 3.2 Feature Engineering

We constructed 23 additional features from domain knowledge: ratio features (e.g., `conductivity_solids_ratio`), interaction terms (e.g., `chloramines_ph_interaction`), additive composites (e.g., `disinfection_stress = Sulfate + Chloramines`), and binary risk flags based on WHO guidelines. RFECV then selected the subset of these 32 features that improved cross-validated ROC-AUC.

These 32 features are materialised into a small file-based feature store under `data/04_feature/` (`mlops_project.feature_store`). A single registry defines every feature's name, dtype, group, and description, and each materialised feature set is written with a content-addressable metadata sidecar recording a schema version and a data-snapshot version. This gives the engineered features a versioned, self-describing contract that training, drift detection, and the test suite all read back through one retrieval API, while the leakage-sensitive learned preprocessing stays out of the store and is re-fitted per fold.

### 3.3 Model Comparison

All models were evaluated with 5-fold stratified cross-validation on the training set and a single held-out test evaluation. The table below reports the primary development metric (CV F1) alongside the test ROC-AUC, which is the success criterion defined in Section 1.

| Rank | Model | CV F1 ± std | Test ROC-AUC | Test F1 |
|------|-------|-------------|--------------|---------|
| 1 | **Extra Trees** | **0.498 ± 0.044** | **0.666** | **0.574** |
| 2 | Hist. Gradient Boosting | 0.489 ± 0.043 | 0.682 | 0.574 |
| 3 | Random Forest | 0.485 ± 0.047 | 0.670 | 0.530 |
| 4 | XGBoost | 0.415 ± 0.049 | 0.674 | 0.506 |
| 5 | Logistic Regression | 0.132 ± 0.077 | 0.551 | 0.136 |

Models are selected by the primary development metric (CV F1), which is computed on the training split only, before the test set is touched. Extra Trees achieves the highest CV F1 (0.498) and is therefore the champion that we serve, explain, and monitor for drift; its test ROC-AUC of 0.666 and test F1 of 0.574 clear both success thresholds from Section 1. The top three tree ensembles are statistically very close on CV F1 (0.498 / 0.489 / 0.485, all within one CV standard deviation of about 0.04), and on the test split Hist. Gradient Boosting edges slightly ahead on ROC-AUC (0.682), but model selection is fixed on the development metric, not the holdout. The logistic regression baseline, while stable (low CV std), is far behind the tree models, confirming the relationship between water quality and potability is not well captured by a linear boundary.

It is worth being direct about what "clearing the thresholds" means here: a ROC-AUC of 0.666 and an accuracy of 0.650 are real but modest, closer to "somewhat better than a coin flip" than to a deployable diagnostic tool, and the same pattern holds for every model family we tried, not just ours. This is consistent with how the dataset is known to behave: the Kaggle Water Potability dataset's labels are synthetically generated and only weakly tied to the nine physicochemical features, so even an ideal classifier cannot push performance much further without additional, more informative measurements. We set the thresholds in Section 1 specifically at a level that separates "the model learned a real, non-trivial signal" from "the model is guessing," and Extra Trees clears that bar, but readers should not mistake it for a clinically usable potability test. The honest conclusion is that the production-readiness work in Section 4 (serving, drift monitoring, retraining triggers) is the more transferable outcome of this project than the absolute accuracy of the classifier itself.

![Extra Trees test confusion matrix](../data/08_reporting/extra_trees_test_confusion_matrix.png)

The confusion matrix on the test split shows the model identifies non-potable samples (the majority class) somewhat more reliably than potable ones, which is expected given the class imbalance; with a positive-class recall of 0.60 it still recovers the majority of potable samples.

### 3.4 SHAP Feature Importance (Extra Trees)

SHAP (SHapley Additive exPlanations) attributes each feature's contribution to individual predictions on a theoretically grounded basis. We used TreeExplainer, which computes exact SHAP values for tree models without approximation.

![Extra Trees SHAP feature importance](../data/08_reporting/extra_trees_shap_summary_plot.png)

pH-related features dominate, consistent with domain knowledge: pH is the single most commonly monitored indicator of water safety. The top feature is the engineered `ph_safe_range` flag (whether pH falls in the safe 6.5-8.5 band), followed by the raw `ph` value itself and the `chloramines_ph_interaction` term, which captures the combined effect of chloramine concentration and acidity since both are disinfection-related and interact chemically. That the single most important feature is engineered, together with several others in the top ranks (`chloramines_ph_interaction`, `solids_sulfate_diff`), confirms that the feature engineering step in Section 3.2 added genuine predictive value beyond the nine raw measurements.

---

## 4. Production Implementation Discussion

### 4.1 Technology Choices and Their Advantages

**Kedro** structures the project as a directed acyclic graph of named nodes, with every intermediate dataset catalogued in a single YAML file. This makes it trivial to rerun any subset of the pipeline in isolation (`kedro run --pipeline data_drift`) and ensures every experiment starts from a reproducible state, instead of a series of scripts with implicit, hard-to-audit dependencies.

**MLflow** provides experiment tracking that goes beyond saving the best model: every CV run, Optuna trial, and holdout evaluation is logged with its exact hyperparameters, metrics, and artifacts, creating a full audit trail we could use to roll back to an earlier model.

**Optuna** with the TPE sampler explores the hyperparameter space more efficiently than grid search by building a probabilistic model of which regions tend to produce better results. For Extra Trees we ran 50 trials, improving CV F1 over the default, untuned parameters. Because TPE is sequential and floating-point sensitive, search is isolated in an opt-in `tuning` pipeline that writes the chosen parameters to disk; the default `modeling` pipeline refits from those locked parameters, so it reproduces identical artifacts on every run.

**FastAPI + Docker** separate the prediction API from the training infrastructure: a consumer sends nine water quality measurements as JSON and gets back a prediction and probability, with no Python or Kedro installation required on their end.

### 4.2 Risks and Mitigations

**Data scale.** The pipeline uses Pandas, which loads the entire dataset into memory on a single machine. That is not a problem for this 3,276-row dataset, but if applied to a continuous monitoring system producing millions of samples per day, Pandas would become a bottleneck. Mitigation: move to a distributed backend (Parquet on S3, PySpark or Polars transforms); we estimate roughly three additional weeks of engineering effort.

**Model drift.** Water quality can shift seasonally or after infrastructure events (pipe corrosion, treatment changes). Comparing the training split against the held-out test split with the KS test, as a sanity baseline, flags only 2 of 32 features (about the false-positive rate expected from random sampling): the two splits come from the same distribution, so this confirms the split is sound but is not itself a drift scenario. To see the pipeline react to an actual shift, we simulated a production sample where a treatment plant changes its disinfection process (lower pH, higher chloramines, sulfate, and trihalomethanes) by perturbing the raw measurements in the test split and recomputing the engineered features. Against this simulated sample, the same KS test now flags 17 of 32 features, and the Extra Trees champion's test ROC-AUC drops from 0.666 to 0.564 (accuracy from 0.650 to 0.547). This is the kind of signal that should trigger a retraining run in production; the check should run automatically on a rolling window of incoming data rather than on demand.

**Class imbalance.** The 61/39 split is manageable but not negligible: the logistic regression baseline in particular struggles with recall on the minority (potable) class. A production system would need a deliberate cost-sensitivity analysis (e.g., SMOTE or class-weighted loss) if the cost of false negatives were higher than in this proof of concept.

**Single-model serving.** The API currently serves only the Extra Trees champion; if a future comparison ranked a different model higher, the Dockerfile and model path would need a manual update. A more mature setup would pull the model from the MLflow registry by alias (`models:/water_potability@champion`) so serving always tracks the latest promoted model.

---

## 5. Package List

The project uses Python 3.13 and manages dependencies with uv. The table below lists the key packages and their pinned versions as declared in `pyproject.toml` and resolved in `uv.lock`.

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.13.13 | Runtime |
| kedro | 1.3.1 | Pipeline orchestration |
| pandas | 2.3.3 | Tabular data manipulation |
| scikit-learn | 1.8.0 | Model training, CV, preprocessing |
| scipy | 1.17.1 | KS-test for drift detection |
| numpy | 2.4.4 | Numerical operations |
| xgboost | 3.2.0 | Gradient boosted trees |
| optuna | 4.8.0 | Hyperparameter optimisation |
| mlflow | 3.12.0 | Experiment tracking |
| great-expectations | 1.17.2 | Raw data contract validation |
| shap | 0.52.0 | Feature importance (SHAP values) |
| fastapi | 0.136.1 | REST API for model serving |
| uvicorn | 0.47.0 | ASGI server for FastAPI |
| matplotlib | 3.10.9 | Confusion matrix and SHAP plots |
| kaggle | 2.1.2 | Dataset download |
| pytest | 9.0.3 | Unit and integration testing |
