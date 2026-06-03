# Water Potability MLOps Context

This context describes the project language for validating, preprocessing, and modeling the water potability dataset.

## Language

**Raw Data Contract**:
A fail-fast agreement that the loaded water potability dataset must satisfy before any preprocessing modifies it.
_Avoid_: Soft check, cleanup step

**Water Potability Dataset**:
The labelled water quality observations used to predict whether a sample is potable.
_Avoid_: Water data, raw CSV

**Potability Label**:
The binary target indicating whether a water sample is not potable or potable.
_Avoid_: Class, target, outcome

**Known Nullable Measurement**:
A water quality measurement where missing observations are expected and accepted within agreed limits.
_Avoid_: Bad data, imputation field

**Physically Plausible Range**:
A broad measurement bound that rejects impossible values without encoding the historical dataset minimum or maximum.
_Avoid_: Dataset range, outlier threshold

**Distribution Drift Check**:
A monitoring check that compares current data statistics against a historical baseline.
_Avoid_: Raw data contract check

**Baseline Model**:
The first simple model used to establish reproducible training, evaluation, and experiment tracking before model comparison or tuning.
_Avoid_: Production model, champion model

**Tree-Based Model**:
A model family whose predictions are built from decision-tree structures, including bagging ensembles, extremely randomized trees, histogram gradient boosting, and boosted trees.
_Avoid_: DT model, decision tree model when referring to ensembles or boosting families

**Development Validation Evaluation**:
The evaluation on the validation split used to compare development choices before touching the final holdout split.
_Avoid_: Final score, test result

**Final Holdout Evaluation**:
The one-time reporting evaluation on the test split after preprocessing and model training decisions are fixed.
_Avoid_: Validation score, tuning metric

**Primary Development Metric**:
The main validation metric used to compare development choices; for the first baseline this is validation F1.
_Avoid_: Test metric, final holdout metric

**Hyperparameter Optimization Evaluation**:
A training-set-only cross-validation process used to select model hyperparameters by the Primary Development Metric.
_Avoid_: Test tuning, holdout optimization

**Learned Preprocessing**:
A fold-local transformation stage fitted only on training rows to produce estimator inputs.
_Avoid_: Global preprocessing, preprocessing pipeline

**Model-Ready Feature Contract**:
A fail-fast agreement that transformed estimator input features must satisfy after Learned Preprocessing.
_Avoid_: Raw data contract, drift check

## Relationships

- A **Water Potability Dataset** must satisfy the **Raw Data Contract** before preprocessing begins.
- Each row in the **Water Potability Dataset** has exactly one **Potability Label**.
- A **Raw Data Contract** allows missing values only for **Known Nullable Measurements**.
- A **Raw Data Contract** checks **Physically Plausible Ranges** rather than historical observed ranges.
- A **Distribution Drift Check** is separate from the fail-fast **Raw Data Contract**.
- A **Baseline Model** is assessed through a **Development Validation Evaluation** and a **Final Holdout Evaluation**.
- A **Primary Development Metric** belongs to the **Development Validation Evaluation**, not the **Final Holdout Evaluation**.
- A **Hyperparameter Optimization Evaluation** may use cross-validation folds from the training split, but never the final holdout split.
- **Learned Preprocessing** produces model-ready feature matrices for model training, evaluation, and inference.
- A **Model-Ready Feature Contract** validates estimator inputs after **Learned Preprocessing**.
- A **Model-Ready Feature Contract** is separate from a **Distribution Drift Check**.
- A **Tree-Based Model** candidate is selected through **Development Validation Evaluation**, not by optimizing the **Final Holdout Evaluation**.

## Example dialogue

> **Dev:** "Should missing sulfate values be imputed before validation?"
> **Domain expert:** "No — the **Raw Data Contract** decides whether the **Water Potability Dataset** is acceptable before preprocessing changes it."

> **Dev:** "Can we choose a model because it has the best test F1?"
> **Domain expert:** "No — model choices use the **Primary Development Metric** from the **Development Validation Evaluation**. The **Final Holdout Evaluation** is reported after decisions are fixed."

> **Dev:** "Can Optuna compare RandomForest trials against test F1?"
> **Domain expert:** "No — Optuna performs a **Hyperparameter Optimization Evaluation** on the training split only. Test F1 belongs to the **Final Holdout Evaluation**."

> **Dev:** "Should we validate scaled fold features against the raw water measurement ranges?"
> **Domain expert:** "No — a **Model-Ready Feature Contract** checks estimator input structure after **Learned Preprocessing**, not raw measurement plausibility or drift."

## Flagged ambiguities

- "data quality validation" was resolved to mean a fail-fast **Raw Data Contract** immediately after loading the dataset and before splitting.
- Missingness is not generally acceptable; it is allowed only for known nullable measurements such as pH, sulfate, and trihalomethanes within agreed thresholds.
- The raw schema is strict: the dataset must contain exactly the expected water quality measurements and the potability label, with no extra modelling columns.
- Plausibility checks use broad physical bounds such as pH between 0 and 14 and non-negative concentration-like measurements.
- Statistical drift checks such as mean, standard deviation, quantile, or class-balance thresholds are out of scope for the first fail-fast validation layer.
- The accepted missingness limits are pH at most 16%, sulfate at most 25%, and trihalomethanes at most 6%.
- The raw contract requires at least one row but does not enforce the historical row count.
- Exact duplicate rows do not fail the first raw contract because the dataset has no domain identity column for a water sample.
- The first **Baseline Model** is a LogisticRegression model intended to establish the training and MLflow tracking path, not to perform model selection.
- **Tree-Based Models** may be compared through the same **Hyperparameter Optimization Evaluation** pattern when they share the same development metric boundary.
- Test split metrics are **Final Holdout Evaluation** metrics and must not drive threshold tuning, hyperparameter tuning, or model selection.
- RandomForest hyperparameter tuning is a **Hyperparameter Optimization Evaluation** driven by binary F1 on training-set cross-validation folds.
- ExtraTrees, HistGradientBoosting, and XGBoost belong to **Tree-Based Model** comparison rather than a single decision-tree baseline.
- "preprocessing" is overloaded: the Kedro preprocessing pipeline prepares engineered split artifacts, while **Learned Preprocessing** is fitted inside modeling folds and final model refits.
- Post-scaling validation is a **Model-Ready Feature Contract**, not a **Raw Data Contract** or **Distribution Drift Check**.
