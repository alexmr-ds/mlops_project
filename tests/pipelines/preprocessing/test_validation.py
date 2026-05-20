"""Unit tests for raw data validation."""

import unittest

import numpy as np
import pandas as pd

from mlops_project.pipelines.preprocessing import validation


class RawDataValidationTests(unittest.TestCase):
    """Tests for the fail-fast raw data contract."""

    def test_validate_raw_data_returns_input_dataframe_when_valid(self) -> None:
        data = _valid_raw_data()

        validated_data = validation.validate_raw_data(data)

        self.assertIs(validated_data, data)

    def test_validate_raw_data_allows_expected_nullable_missingness(self) -> None:
        data = _valid_raw_data()
        data.loc[:15, 'ph'] = np.nan
        data.loc[:24, 'Sulfate'] = np.nan
        data.loc[:5, 'Trihalomethanes'] = np.nan

        validated_data = validation.validate_raw_data(data)

        self.assertIs(validated_data, data)

    def test_validate_raw_data_fails_for_missing_columns(self) -> None:
        data = _valid_raw_data().drop(columns=['ph'])

        with self.assertRaisesRegex(ValueError, 'expect_table_columns_to_match_set'):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_extra_columns(self) -> None:
        data = _valid_raw_data()
        data['unexpected_feature'] = 1.0

        with self.assertRaisesRegex(ValueError, 'expect_table_columns_to_match_set'):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_disallowed_nulls(self) -> None:
        data = _valid_raw_data()
        data.loc[0, 'Hardness'] = np.nan

        with self.assertRaisesRegex(ValueError, "column 'Hardness'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_excess_nullable_missingness(self) -> None:
        data = _valid_raw_data()
        data.loc[:16, 'ph'] = np.nan

        with self.assertRaisesRegex(ValueError, "column 'ph'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_non_numeric_types(self) -> None:
        data = _valid_raw_data()
        data['Conductivity'] = data['Conductivity'].astype(str)

        with self.assertRaisesRegex(ValueError, "column 'Conductivity'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_invalid_potability_values(self) -> None:
        data = _valid_raw_data()
        data.loc[0, 'Potability'] = 2

        with self.assertRaisesRegex(ValueError, "column 'Potability'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_invalid_ph_and_negative_values(self) -> None:
        data = _valid_raw_data()
        data.loc[0, 'ph'] = 15.0
        data.loc[1, 'Solids'] = -1.0

        with self.assertRaisesRegex(ValueError, "column 'ph'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_empty_dataset(self) -> None:
        data = _valid_raw_data().iloc[0:0].copy()

        with self.assertRaisesRegex(ValueError, 'expect_table_row_count_to_be_between'):
            validation.validate_raw_data(data)

    def test_validate_raw_data_allows_duplicate_rows(self) -> None:
        data = _valid_raw_data()
        duplicate_data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

        validated_data = validation.validate_raw_data(duplicate_data)

        self.assertIs(validated_data, duplicate_data)


def _valid_raw_data(row_count: int = 100) -> pd.DataFrame:
    """Return a structurally valid raw water potability dataframe."""
    return pd.DataFrame(
        {
            'ph': np.linspace(6.5, 8.5, row_count),
            'Hardness': np.linspace(150.0, 250.0, row_count),
            'Solids': np.linspace(10_000.0, 30_000.0, row_count),
            'Chloramines': np.linspace(4.0, 9.0, row_count),
            'Sulfate': np.linspace(250.0, 380.0, row_count),
            'Conductivity': np.linspace(300.0, 500.0, row_count),
            'Organic_carbon': np.linspace(8.0, 20.0, row_count),
            'Trihalomethanes': np.linspace(40.0, 90.0, row_count),
            'Turbidity': np.linspace(2.0, 5.0, row_count),
            'Potability': np.resize([0, 1], row_count),
        }
    )


if __name__ == '__main__':
    unittest.main()
