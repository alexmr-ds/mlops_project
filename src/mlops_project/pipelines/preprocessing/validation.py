"""Raw data validation for the preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import great_expectations as gx
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
KNOWN_NULLABLE_MISSINGNESS_LIMITS = {
    "ph": 0.16,
    "Sulfate": 0.25,
    "Trihalomethanes": 0.06,
}
NUMERIC_TYPE_NAMES = [
    "float64",
    "float32",
    "float16",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "Int64",
    "Int32",
    "Int16",
    "Int8",
]
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
MAX_FAILURE_LINES = 12


def validate_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw water potability dataset and return it unchanged."""
    context = _get_validation_context()
    suite = build_raw_data_expectation_suite()
    validation_result = _validate_dataframe(data, suite, context)

    if not validation_result.success:
        raise ValueError(_build_validation_error_message(validation_result))

    return data


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
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column,
                type_list=NUMERIC_TYPE_NAMES,
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


def _build_validation_error_message(validation_result: Any) -> str:
    """Return a compact failure summary for a Great Expectations result."""
    failed_results = [
        result for result in validation_result.results if not bool(result.success)
    ]
    failure_lines = [
        _summarize_failed_expectation(result)
        for result in failed_results[:MAX_FAILURE_LINES]
    ]
    remaining_failures = len(failed_results) - len(failure_lines)
    if remaining_failures > 0:
        failure_lines.append(f"...and {remaining_failures} more failed expectation(s)")

    statistics = validation_result.statistics
    return (
        "Raw data validation failed: "
        f"{statistics['unsuccessful_expectations']} of "
        f"{statistics['evaluated_expectations']} expectation(s) failed. "
        + "; ".join(failure_lines)
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
