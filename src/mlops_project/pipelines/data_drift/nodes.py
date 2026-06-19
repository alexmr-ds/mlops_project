"""Node functions for data drift detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


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
