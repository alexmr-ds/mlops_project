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

    For each feature column, we run a two-sample Kolmogorov-Smirnov test comparing
    the training distribution (our reference baseline) against the test distribution
    (representing data the model would see in production).  A low p-value indicates
    that the two samples are unlikely to come from the same distribution, which is a
    signal that drift may have occurred.

    Returns a report with one row per feature containing the KS statistic, p-value,
    and a boolean flag indicating whether the feature is considered drifted at the
    configured significance level.
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

    X_train and X_test in this project come from the same random split of the
    same dataset, so comparing them with the KS test mostly checks that the split
    was done correctly -- it is not a real drift scenario. To exercise the drift
    detector and the model the way they would actually be used in production, we
    apply a hypothetical but realistic shift (e.g. a treatment plant changing its
    disinfection process) to the raw measurements and recompute the engineered
    features from the perturbed values, exactly as `_engineer_feature_frame` does
    during training.
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

    Reuses the same metric calculation as the regular test evaluation
    (`evaluation.evaluate_model`) so the resulting row can be compared directly
    against e.g. `random_forest_test_metrics` to see how far performance drops
    once the input distribution no longer matches what the model was trained on.
    """
    metrics_frame, _ = modeling_evaluation.evaluate_model(model, simulated_X_test, y_test)
    return metrics_frame
