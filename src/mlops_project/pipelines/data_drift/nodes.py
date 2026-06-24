"""Node functions for data drift detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from mlops_project.modeling import evaluation as modeling_evaluation
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame

_RAW_MEASUREMENT_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]


def detect_feature_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Detect distribution shift between training and test features using the KS test.

    Runs a two-sample Kolmogorov-Smirnov test per feature column, comparing the
    training distribution (the reference baseline) against the test distribution
    (the data the model would see in production). A low p-value means the two
    samples probably aren't from the same distribution, which is a sign of drift.

    Returns one row per feature with the KS statistic, p-value, and a boolean
    drifted flag based on the configured significance level.
    """
    threshold = float(parameters.get("significance_threshold", 0.05))

    shared_columns = [col for col in X_train.columns if col in X_test.columns]
    rows = []
    for feature in shared_columns:
        train_values = X_train[feature].dropna().to_numpy()
        test_values = X_test[feature].dropna().to_numpy()
        ks_result = stats.ks_2samp(train_values, test_values)
        rows.append(
            {
                "feature": feature,
                "ks_statistic": round(float(ks_result.statistic), 6),
                "p_value": round(float(ks_result.pvalue), 6),
                "drifted": bool(ks_result.pvalue < threshold),
            }
        )

    report = pd.DataFrame(rows, columns=["feature", "ks_statistic", "p_value", "drifted"])
    return report


def simulate_production_drift(
    X_test: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Simulate a production sample whose raw measurements have shifted.

    X_train and X_test come from the same random split of the same dataset, so
    comparing them with the KS test mostly just checks that the split was done
    correctly -- not a real drift scenario. This applies a hypothetical but
    realistic shift to the raw measurements (e.g. a treatment plant changing its
    disinfection process) and recomputes the engineered features from the
    perturbed values, the same way `_engineer_feature_frame` does during
    training, so the drift detector and the model can be tested the way they'd
    actually be used in production.
    """
    shifts = parameters.get("simulated_shift", {})

    perturbed_raw = X_test[_RAW_MEASUREMENT_COLUMNS].copy()
    for column, shift_amount in shifts.items():
        if column in perturbed_raw.columns:
            perturbed_raw[column] = perturbed_raw[column] + float(shift_amount)

    simulated = _engineer_feature_frame(perturbed_raw)
    return simulated.reindex(columns=X_test.columns)


def evaluate_model_under_simulated_drift(
    model: Any,
    simulated_X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Score the trained model on the simulated production sample.

    Reuses the regular test evaluation (`evaluation.evaluate_model`), so the
    output row is directly comparable to e.g. `random_forest_test_metrics` --
    that comparison is what shows how far performance drops once the input
    distribution no longer matches what the model was trained on.
    """
    metrics_frame, _ = modeling_evaluation.evaluate_model(model, simulated_X_test, y_test)
    return metrics_frame
