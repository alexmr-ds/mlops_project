"""Model-ready validation contracts for learned preprocessing outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import great_expectations as gx
import numpy as np
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

MODEL_READY_NUMERIC_TYPES = [
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
MODEL_READY_LABEL_VALUES = [0, 1]
MAX_FAILURE_LINES = 12


def validate_model_ready_features(
    features: pd.DataFrame,
    expected_columns: Sequence[str],
    artifact_name: str,
) -> pd.DataFrame:
    """Validate transformed model-ready features and return them unchanged."""
    column_list = _validate_feature_contract_inputs(
        features,
        expected_columns,
        artifact_name,
    )
    context = _get_validation_context()
    suite = build_model_ready_feature_expectation_suite(column_list)
    validation_result = _validate_dataframe(features, suite, context)
    finite_failures = _summarize_finite_value_failures(
        features,
        column_list,
        artifact_name,
    )

    if not validation_result.success or finite_failures:
        raise ValueError(
            _build_model_ready_feature_error_message(
                artifact_name,
                validation_result,
                extra_failure_lines=finite_failures,
            )
        )

    return features


def validate_model_ready_training_artifacts(
    features: pd.DataFrame,
    labels: pd.Series,
    expected_columns: Sequence[str],
    artifact_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate transformed training features and labels after row filtering."""
    validated_features = validate_model_ready_features(
        features,
        expected_columns,
        f"{artifact_name} features",
    )
    label_failures = _summarize_label_failures(
        labels=labels,
        features=validated_features,
        artifact_name=f"{artifact_name} labels",
    )
    if label_failures:
        raise ValueError(
            _build_model_ready_artifact_error_message(
                artifact_name,
                label_failures,
            )
        )

    return validated_features, labels


def build_model_ready_feature_expectation_suite(
    expected_columns: Sequence[str],
) -> gx.ExpectationSuite:
    """Build the model-ready feature expectation suite."""
    column_list = list(expected_columns)
    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1),
        gx.expectations.ExpectTableColumnsToMatchOrderedList(column_list=column_list),
    ]

    for column in column_list:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column,
                type_list=MODEL_READY_NUMERIC_TYPES,
            )
        )
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    return gx.ExpectationSuite(
        name="model_ready_feature_contract",
        expectations=expectations,
    )


def _validate_feature_contract_inputs(
    features: pd.DataFrame,
    expected_columns: Sequence[str],
    artifact_name: str,
) -> list[str]:
    """Validate inputs before constructing the Great Expectations suite."""
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("artifact_name must be a non-empty string.")
    if not isinstance(features, pd.DataFrame):
        raise ValueError(f"{artifact_name} must be a pandas DataFrame.")
    if isinstance(expected_columns, str):
        raise ValueError("expected_columns must be a non-empty sequence of strings.")
    try:
        column_list = list(expected_columns)
    except TypeError as error:
        raise ValueError(
            "expected_columns must be a non-empty sequence of strings."
        ) from error
    if not column_list or not all(isinstance(column, str) for column in column_list):
        raise ValueError("expected_columns must be a non-empty sequence of strings.")
    if len(set(column_list)) != len(column_list):
        raise ValueError("expected_columns must not contain duplicate columns.")
    return column_list


def _validate_dataframe(
    data: pd.DataFrame, suite: gx.ExpectationSuite, context: Any
) -> Any:
    """Run a Great Expectations suite against an in-memory dataframe."""
    batch = context.data_sources.pandas_default.read_dataframe(dataframe=data)
    return batch.validate(suite, result_format="BASIC")


def _get_validation_context() -> Any:
    """Return an ephemeral Great Expectations context for model validation."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(
        globally=False,
        metric_calculations=False,
    )
    return context


def _summarize_finite_value_failures(
    features: pd.DataFrame,
    expected_columns: list[str],
    artifact_name: str,
) -> list[str]:
    """Return failures for infinite model-ready feature values."""
    if features.columns.tolist() != expected_columns:
        return []

    try:
        values = features.loc[:, expected_columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return [f"{artifact_name} contains non-numeric feature values"]

    finite_or_missing = np.isfinite(values) | np.isnan(values)
    if finite_or_missing.all():
        return []
    return [f"{artifact_name} contains infinite feature values"]


def _summarize_label_failures(
    labels: pd.Series,
    features: pd.DataFrame,
    artifact_name: str,
) -> list[str]:
    """Return failures for model-ready labels after training row filtering."""
    if not isinstance(labels, pd.Series):
        return [f"{artifact_name} must be a pandas Series"]

    failures = []
    if labels.empty:
        failures.append(f"{artifact_name} must contain at least one label")
    if len(features) != len(labels):
        failures.append(
            f"{artifact_name} row count must match features "
            f"(features={len(features)}, labels={len(labels)})"
        )
    if not features.index.equals(labels.index):
        failures.append(f"{artifact_name} index must match features")
    if labels.isna().any():
        failures.append(f"{artifact_name} contains null label values")
    if not labels.isin(MODEL_READY_LABEL_VALUES).all():
        failures.append(
            f"{artifact_name} must contain only binary labels "
            f"{MODEL_READY_LABEL_VALUES}"
        )
    return failures


def _build_model_ready_feature_error_message(
    artifact_name: str,
    validation_result: Any,
    extra_failure_lines: list[str] | None = None,
) -> str:
    """Return a compact failure summary for model-ready feature validation."""
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

    return (
        f"{artifact_name} model-ready feature validation failed: "
        f"{len(failed_results) + len(extra_failure_lines or [])} failure(s). "
        + "; ".join(failure_lines)
    )


def _build_model_ready_artifact_error_message(
    artifact_name: str,
    failure_lines: list[str],
) -> str:
    """Return a compact failure summary for model-ready artifact validation."""
    visible_failure_lines = failure_lines[:MAX_FAILURE_LINES]
    remaining_failures = len(failure_lines) - len(visible_failure_lines)
    if remaining_failures > 0:
        visible_failure_lines.append(
            f"...and {remaining_failures} more failed expectation(s)"
        )

    return (
        f"{artifact_name} model-ready artifact validation failed: "
        f"{len(failure_lines)} failure(s). "
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
