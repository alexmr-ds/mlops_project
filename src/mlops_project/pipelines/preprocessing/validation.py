"""Great Expectations validation for preprocessing pipeline artifacts."""

from __future__ import annotations

from typing import Any

import great_expectations as gx
import numpy as np
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

EXPECTED_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Potability",
]
EXPECTED_DTYPES = {
    "ph": "float64",
    "Hardness": "float64",
    "Solids": "float64",
    "Chloramines": "float64",
    "Sulfate": "float64",
    "Conductivity": "float64",
    "Organic_carbon": "float64",
    "Trihalomethanes": "float64",
    "Turbidity": "float64",
    "Potability": "int64",
}
KNOWN_NULLABLE_MISSINGNESS_LIMITS = {
    "ph": 0.16,
    "Sulfate": 0.25,
    "Trihalomethanes": 0.06,
}
NON_NEGATIVE_MEASUREMENT_COLUMNS = [
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
PREPROCESSED_LABEL_DTYPE = "int64"
MAX_FAILURE_LINES = 12


def validate_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw water potability dataset and return it unchanged."""
    context = _get_validation_context()
    suite = build_raw_data_expectation_suite()
    validation_result = _validate_dataframe(data, suite, context)
    dtype_failures = _summarize_dtype_mismatches(data)

    if not validation_result.success or dtype_failures:
        raise ValueError(
            _build_validation_error_message(
                validation_result,
                extra_failure_lines=dtype_failures,
            )
        )

    return data


def validate_modeling_input_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Validate engineered train/test artifacts for downstream modeling."""
    failures = []
    failures.extend(_summarize_modeling_feature_failures("X_train", X_train))
    failures.extend(_summarize_modeling_feature_failures("X_test", X_test))
    failures.extend(
        _summarize_preprocessed_label_failures(
            artifact_name="y_train",
            labels=y_train,
            required_non_empty=True,
        )
    )
    failures.extend(
        _summarize_preprocessed_label_failures(
            artifact_name="y_test",
            labels=y_test,
            required_non_empty=True,
        )
    )
    failures.extend(_summarize_split_alignment_failures("train", X_train, y_train))
    failures.extend(_summarize_split_alignment_failures("test", X_test, y_test))

    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if X_train.columns.tolist() != X_test.columns.tolist():
            failures.append("X_train and X_test columns must match in order")

    if failures:
        raise ValueError(_build_modeling_input_validation_error_message(failures))

    return X_train, X_test, y_train, y_test


def build_raw_data_expectation_suite() -> gx.ExpectationSuite:
    """Build the raw water potability expectation suite."""
    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1),
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS,
            exact_match=True,
        ),
    ]

    for column in EXPECTED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_=EXPECTED_DTYPES[column],
            )
        )

    required_columns = sorted(
        set(EXPECTED_COLUMNS) - set(KNOWN_NULLABLE_MISSINGNESS_LIMITS)
    )
    for column in required_columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    for column, max_missing_fraction in KNOWN_NULLABLE_MISSINGNESS_LIMITS.items():
        expectations.append(
            gx.expectations.ExpectColumnProportionOfNonNullValuesToBeBetween(
                column=column,
                min_value=1 - max_missing_fraction,
            )
        )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ph",
            min_value=0,
            max_value=14,
        )
    )
    for column in NON_NEGATIVE_MEASUREMENT_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )

    expectations.append(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Potability",
            value_set=[0, 1],
        )
    )
    return gx.ExpectationSuite(
        name="water_potability_raw_data_contract",
        expectations=expectations,
    )


def _validate_dataframe(
    data: pd.DataFrame, suite: gx.ExpectationSuite, context: Any
) -> Any:
    """Run a Great Expectations suite against an in-memory dataframe."""
    batch = context.data_sources.pandas_default.read_dataframe(dataframe=data)
    return batch.validate(suite, result_format="BASIC")


def _get_validation_context() -> Any:
    """Return an ephemeral Great Expectations context configured for pipeline use."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(
        globally=False,
        metric_calculations=False,
    )
    return context


def _build_validation_error_message(
    validation_result: Any, extra_failure_lines: list[str] | None = None
) -> str:
    """Return a compact failure summary for a Great Expectations result."""
    failed_results = [
        result for result in validation_result.results if not bool(result.success)
    ]
    failure_lines = [
        _summarize_failed_expectation(result)
        for result in failed_results[:MAX_FAILURE_LINES]
    ]
    remaining_slots = 0
    if extra_failure_lines:
        remaining_slots = max(MAX_FAILURE_LINES - len(failure_lines), 0)
        failure_lines.extend(extra_failure_lines[:remaining_slots])

    remaining_failures = len(failed_results) - len(failure_lines)
    if extra_failure_lines:
        remaining_failures += max(len(extra_failure_lines) - remaining_slots, 0)
    if remaining_failures > 0:
        failure_lines.append(f"...and {remaining_failures} more failed expectation(s)")

    statistics = validation_result.statistics
    return (
        "Raw data validation failed: "
        f"{statistics['unsuccessful_expectations'] + len(extra_failure_lines or [])} of "
        f"{statistics['evaluated_expectations']} expectation(s) failed. "
        + "; ".join(failure_lines)
    )


def _summarize_dtype_mismatches(data: pd.DataFrame) -> list[str]:
    """Return exact pandas dtype mismatches for columns present in the dataframe."""
    failures = []
    for column, expected_dtype in EXPECTED_DTYPES.items():
        if column not in data.columns:
            continue

        observed_dtype = str(data.dtypes[column])
        if observed_dtype != expected_dtype:
            failures.append(
                "expect_column_values_to_be_of_type "
                f"for column '{column}' "
                f"(observed={observed_dtype!r}, expected={expected_dtype!r})"
            )
    return failures


def _summarize_modeling_feature_failures(
    artifact_name: str, feature_frame: pd.DataFrame
) -> list[str]:
    """Return failures for engineered feature matrices before learned preprocessing."""
    if not isinstance(feature_frame, pd.DataFrame):
        return [f"{artifact_name} must be a pandas DataFrame"]

    failures = []
    if feature_frame.empty:
        failures.append(f"{artifact_name} must contain at least one row")
    if len(feature_frame.columns) == 0:
        failures.append(f"{artifact_name} must contain at least one feature")

    non_numeric_columns = [
        column
        for column in feature_frame.columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric_columns:
        failures.append(
            f"{artifact_name} columns must be numeric: {non_numeric_columns}"
        )

    if feature_frame.empty:
        return failures

    try:
        values = feature_frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        failures.append(f"{artifact_name} contains non-numeric feature values")
    else:
        finite_or_missing = np.isfinite(values) | np.isnan(values)
        if not finite_or_missing.all():
            failures.append(f"{artifact_name} contains non-finite feature values")

    return failures


def _summarize_preprocessed_label_failures(
    artifact_name: str,
    labels: pd.Series,
    required_non_empty: bool,
) -> list[str]:
    """Return exact pandas failures for one final label series."""
    failures = []
    if required_non_empty and labels.empty:
        failures.append(f"{artifact_name} must contain at least one label")

    observed_dtype = str(labels.dtype)
    if observed_dtype != PREPROCESSED_LABEL_DTYPE:
        failures.append(
            f"{artifact_name} must be {PREPROCESSED_LABEL_DTYPE}, "
            f"observed {observed_dtype!r}"
        )
    if labels.isna().any():
        failures.append(f"{artifact_name} contains null label values")
    if not labels.isin([0, 1]).all():
        failures.append(f"{artifact_name} must contain only binary labels 0 or 1")
    return failures


def _summarize_split_alignment_failures(
    split_name: str, features: pd.DataFrame, labels: pd.Series
) -> list[str]:
    """Return row-count and index alignment failures for one split."""
    if not isinstance(features, pd.DataFrame) or not isinstance(labels, pd.Series):
        return []

    failures = []
    if len(features) != len(labels):
        failures.append(
            f"{split_name} split X/y row counts must match "
            f"(features={len(features)}, labels={len(labels)})"
        )
    if not features.index.equals(labels.index):
        failures.append(f"{split_name} split X/y indexes must match")
    return failures


def _build_modeling_input_validation_error_message(failure_lines: list[str]) -> str:
    """Return a compact failure summary for modeling input validation."""
    visible_failure_lines = failure_lines[:MAX_FAILURE_LINES]
    remaining_failures = len(failure_lines) - len(visible_failure_lines)
    if remaining_failures > 0:
        visible_failure_lines.append(
            f"...and {remaining_failures} more failed expectation(s)"
        )

    return (
        f"Modeling input validation failed: {len(failure_lines)} failure(s). "
        + "; ".join(visible_failure_lines)
    )


def _summarize_failed_expectation(result: Any) -> str:
    """Summarize one failed Great Expectations expectation."""
    expectation_config = result.expectation_config
    expectation_type = expectation_config.type
    kwargs = {
        key: value
        for key, value in expectation_config.kwargs.items()
        if key != "batch_id"
    }
    scope = f" for column '{kwargs['column']}'" if "column" in kwargs else ""
    details = _summarize_result_details(result.result)
    if result.exception_info and result.exception_info.get("raised_exception"):
        details.append(f"exception={result.exception_info.get('exception_message')!r}")

    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{expectation_type}{scope}{detail_text}"


def _summarize_result_details(result_details: dict[str, Any]) -> list[str]:
    """Extract stable, concise details from a Great Expectations result dict."""
    details = []
    if "observed_value" in result_details:
        details.append(f"observed={result_details['observed_value']!r}")
    if "unexpected_count" in result_details:
        details.append(f"unexpected_count={result_details['unexpected_count']}")
    if "unexpected_percent" in result_details:
        details.append(
            f"unexpected_percent={result_details['unexpected_percent']:.2f}"
        )
    partial_unexpected = result_details.get("partial_unexpected_list")
    if partial_unexpected:
        details.append(f"examples={partial_unexpected[:5]!r}")
    return details
