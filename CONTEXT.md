# Water Potability MLOps Context

This context describes the project language for validating and preprocessing the water potability dataset before model training.

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

## Relationships

- A **Water Potability Dataset** must satisfy the **Raw Data Contract** before preprocessing begins.
- Each row in the **Water Potability Dataset** has exactly one **Potability Label**.
- A **Raw Data Contract** allows missing values only for **Known Nullable Measurements**.
- A **Raw Data Contract** checks **Physically Plausible Ranges** rather than historical observed ranges.
- A **Distribution Drift Check** is separate from the fail-fast **Raw Data Contract**.

## Example dialogue

> **Dev:** "Should missing sulfate values be imputed before validation?"
> **Domain expert:** "No — the **Raw Data Contract** decides whether the **Water Potability Dataset** is acceptable before preprocessing changes it."

## Flagged ambiguities

- "data quality validation" was resolved to mean a fail-fast **Raw Data Contract** immediately after loading the dataset and before splitting.
- Missingness is not generally acceptable; it is allowed only for known nullable measurements such as pH, sulfate, and trihalomethanes within agreed thresholds.
- The raw schema is strict: the dataset must contain exactly the expected water quality measurements and the potability label, with no extra modelling columns.
- Plausibility checks use broad physical bounds such as pH between 0 and 14 and non-negative concentration-like measurements.
- Statistical drift checks such as mean, standard deviation, quantile, or class-balance thresholds are out of scope for the first fail-fast validation layer.
- The accepted missingness limits are pH at most 16%, sulfate at most 25%, and trihalomethanes at most 6%.
- The raw contract requires at least one row but does not enforce the historical row count.
- Exact duplicate rows do not fail the first raw contract because the dataset has no domain identity column for a water sample.
