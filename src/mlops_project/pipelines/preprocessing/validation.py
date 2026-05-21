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
PREPROCESSED_FEATURE_DTYPE = "float64"
PREPROCESSED_LABEL_DTYPE = "int64"
PREPROCESSED_LABEL_COLUMN = "label"
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


def validate_preprocessed_data(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    selected_features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    list[str],
]:
    """Validate final model-ready preprocessing artifacts and return them unchanged."""
    context = _get_validation_context()
    failures = _summarize_selected_features_failures(selected_features)
    expected_columns = (
        selected_features if _is_valid_selected_features(selected_features) else []
    )

    feature_splits = [
        ("X_train", X_train, True),
        ("X_validation", X_validation, False),
        ("X_test", X_test, True),
    ]
    label_splits = [
        ("y_train", y_train, True),
        ("y_validation", y_validation, False),
        ("y_test", y_test, True),
    ]

    for artifact_name, feature_frame, required_non_empty in feature_splits:
        failures.extend(
            _summarize_feature_artifact_failures(
                artifact_name=artifact_name,
                feature_frame=feature_frame,
                selected_features=expected_columns,
                required_non_empty=required_non_empty,
                context=context,
            )
        )

    for artifact_name, labels, required_non_empty in label_splits:
        failures.extend(
            _summarize_label_artifact_failures(
                artifact_name=artifact_name,
                labels=labels,
                required_non_empty=required_non_empty,
                context=context,
            )
        )

    failures.extend(_summarize_split_alignment_failures("train", X_train, y_train))
    failures.extend(
        _summarize_split_alignment_failures("validation", X_validation, y_validation)
    )
    failures.extend(_summarize_split_alignment_failures("test", X_test, y_test))

    if failures:
        raise ValueError(_build_preprocessed_validation_error_message(failures))

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        selected_features,
    )


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


def build_preprocessed_feature_expectation_suite(
    artifact_name: str, selected_features: list[str], min_row_count: int
) -> gx.ExpectationSuite:
    """Build a final feature-matrix expectation suite."""
    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=min_row_count),
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=selected_features,
            exact_match=True,
        ),
    ]

    for column in selected_features:
        expectations.extend(
            [
                gx.expectations.ExpectColumnValuesToBeOfType(
                    column=column,
                    type_=PREPROCESSED_FEATURE_DTYPE,
                ),
                gx.expectations.ExpectColumnValuesToNotBeNull(column=column),
            ]
        )

    return gx.ExpectationSuite(
        name=f"{artifact_name}_preprocessed_feature_contract",
        expectations=expectations,
    )


def build_preprocessed_label_expectation_suite(
    artifact_name: str, min_row_count: int
) -> gx.ExpectationSuite:
    """Build a final label-vector expectation suite."""
    return gx.ExpectationSuite(
        name=f"{artifact_name}_preprocessed_label_contract",
        expectations=[
            gx.expectations.ExpectTableRowCountToBeBetween(min_value=min_row_count),
            gx.expectations.ExpectTableColumnsToMatchSet(
                column_set=[PREPROCESSED_LABEL_COLUMN],
                exact_match=True,
            ),
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=PREPROCESSED_LABEL_COLUMN,
                type_=PREPROCESSED_LABEL_DTYPE,
            ),
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=PREPROCESSED_LABEL_COLUMN,
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=PREPROCESSED_LABEL_COLUMN,
                value_set=[0, 1],
            ),
        ],
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


def _summarize_feature_artifact_failures(
    artifact_name: str,
    feature_frame: pd.DataFrame,
    selected_features: list[str],
    required_non_empty: bool,
    context: Any,
) -> list[str]:
    """Return GE and pandas failures for one final feature dataframe."""
    if not isinstance(feature_frame, pd.DataFrame):
        return [f"{artifact_name} must be a pandas DataFrame"]

    min_row_count = 1 if required_non_empty else 0
    suite = build_preprocessed_feature_expectation_suite(
        artifact_name=artifact_name,
        selected_features=selected_features,
        min_row_count=min_row_count,
    )
    validation_result = _validate_dataframe(feature_frame, suite, context)
    failures = _summarize_validation_result_failures(
        artifact_name,
        validation_result,
    )
    failures.extend(
        _summarize_preprocessed_feature_frame_failures(
            artifact_name=artifact_name,
            feature_frame=feature_frame,
            selected_features=selected_features,
            required_non_empty=required_non_empty,
        )
    )
    return failures


def _summarize_label_artifact_failures(
    artifact_name: str,
    labels: pd.Series,
    required_non_empty: bool,
    context: Any,
) -> list[str]:
    """Return GE and pandas failures for one final label series."""
    if not isinstance(labels, pd.Series):
        return [f"{artifact_name} must be a pandas Series"]

    min_row_count = 1 if required_non_empty else 0
    label_frame = labels.rename(PREPROCESSED_LABEL_COLUMN).to_frame()
    suite = build_preprocessed_label_expectation_suite(
        artifact_name=artifact_name,
        min_row_count=min_row_count,
    )
    validation_result = _validate_dataframe(label_frame, suite, context)
    failures = _summarize_validation_result_failures(
        artifact_name,
        validation_result,
    )
    failures.extend(
        _summarize_preprocessed_label_failures(
            artifact_name=artifact_name,
            labels=labels,
            required_non_empty=required_non_empty,
        )
    )
    return failures


def _summarize_selected_features_failures(selected_features: list[str]) -> list[str]:
    """Return failures for the selected feature schema object."""
    if not isinstance(selected_features, list):
        return ["selected_features must be a list of unique feature names"]

    failures = []
    if not selected_features:
        failures.append("selected_features must contain at least one feature")
    if not all(isinstance(feature, str) for feature in selected_features):
        failures.append("selected_features must contain only strings")
    if all(isinstance(feature, str) for feature in selected_features) and len(
        set(selected_features)
    ) != len(selected_features):
        failures.append("selected_features must not contain duplicate feature names")
    return failures


def _is_valid_selected_features(selected_features: list[str]) -> bool:
    """Return whether selected_features can be used as an expected schema."""
    return not _summarize_selected_features_failures(selected_features)


def _summarize_preprocessed_feature_frame_failures(
    artifact_name: str,
    feature_frame: pd.DataFrame,
    selected_features: list[str],
    required_non_empty: bool,
) -> list[str]:
    """Return exact pandas failures for one final feature dataframe."""
    failures = []
    if required_non_empty and feature_frame.empty:
        failures.append(f"{artifact_name} must contain at least one row")

    if feature_frame.columns.tolist() != selected_features:
        failures.append(
            f"{artifact_name} columns must match selected_features in order"
        )

    for column in feature_frame.columns:
        observed_dtype = str(feature_frame.dtypes[column])
        if observed_dtype != PREPROCESSED_FEATURE_DTYPE:
            failures.append(
                f"{artifact_name} column '{column}' must be "
                f"{PREPROCESSED_FEATURE_DTYPE}, observed {observed_dtype!r}"
            )

    if feature_frame.isna().any().any():
        failures.append(f"{artifact_name} contains null feature values")

    if feature_frame.empty:
        return failures

    try:
        finite_values = np.isfinite(feature_frame.to_numpy(dtype=float))
    except (TypeError, ValueError):
        failures.append(f"{artifact_name} contains non-numeric feature values")
    else:
        if not finite_values.all():
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


def _summarize_validation_result_failures(
    artifact_name: str, validation_result: Any
) -> list[str]:
    """Return compact GE failures for one named artifact."""
    return [
        f"{artifact_name}: {_summarize_failed_expectation(result)}"
        for result in validation_result.results
        if not bool(result.success)
    ]


def _build_preprocessed_validation_error_message(failure_lines: list[str]) -> str:
    """Return a compact failure summary for final preprocessing validation."""
    visible_failure_lines = failure_lines[:MAX_FAILURE_LINES]
    remaining_failures = len(failure_lines) - len(visible_failure_lines)
    if remaining_failures > 0:
        visible_failure_lines.append(
            f"...and {remaining_failures} more failed expectation(s)"
        )

    return (
        f"Preprocessed data validation failed: {len(failure_lines)} failure(s). "
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
