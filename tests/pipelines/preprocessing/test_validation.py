"""Unit tests for preprocessing data validation."""

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

    def test_validate_raw_data_fails_for_float32_feature_dtype(self) -> None:
        data = _valid_raw_data()
        data['Conductivity'] = data['Conductivity'].astype(np.float32)

        with self.assertRaisesRegex(ValueError, "column 'Conductivity'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_integer_feature_dtype(self) -> None:
        data = _valid_raw_data()
        data['Conductivity'] = np.arange(len(data), dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "column 'Conductivity'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_invalid_potability_values(self) -> None:
        data = _valid_raw_data()
        data.loc[0, 'Potability'] = 2

        with self.assertRaisesRegex(ValueError, "column 'Potability'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_float_potability_dtype(self) -> None:
        data = _valid_raw_data()
        data['Potability'] = data['Potability'].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "column 'Potability'"):
            validation.validate_raw_data(data)

    def test_validate_raw_data_fails_for_nullable_integer_potability_dtype(self) -> None:
        data = _valid_raw_data()
        data['Potability'] = pd.Series(data['Potability'], dtype='Int64')

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


class PreprocessedDataValidationTests(unittest.TestCase):
    """Tests for the fail-fast final preprocessing contract."""

    def test_validate_preprocessed_data_returns_inputs_when_valid(self) -> None:
        artifacts = _valid_preprocessed_artifacts()

        validated_artifacts = validation.validate_preprocessed_data(*artifacts)

        for validated_artifact, artifact in zip(validated_artifacts, artifacts):
            self.assertIs(validated_artifact, artifact)

    def test_validate_preprocessed_data_allows_structured_empty_validation_split(
        self,
    ) -> None:
        artifacts = _valid_preprocessed_artifacts(validation_rows=0)

        validated_artifacts = validation.validate_preprocessed_data(*artifacts)

        self.assertIs(validated_artifacts[1], artifacts[1])
        self.assertTrue(validated_artifacts[1].empty)
        self.assertListEqual(validated_artifacts[1].columns.tolist(), artifacts[6])
        self.assertTrue((validated_artifacts[1].dtypes == np.float64).all())

    def test_validate_preprocessed_data_fails_for_empty_train_or_test(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        X_train = X_train.iloc[0:0].copy()
        y_train = y_train.iloc[0:0].copy()

        with self.assertRaisesRegex(ValueError, 'X_train must contain at least one row'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_mismatched_index(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        y_train.index = [100, 101, 102]

        with self.assertRaisesRegex(ValueError, 'train split X/y indexes must match'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_wrong_feature_order(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        X_test = X_test[['feature_b', 'feature_a']]

        with self.assertRaisesRegex(ValueError, 'X_test columns'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_float32_feature_dtype(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        X_train['feature_a'] = X_train['feature_a'].astype(np.float32)

        with self.assertRaisesRegex(ValueError, "column 'feature_a'"):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_null_feature_values(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        X_train.loc[10, 'feature_a'] = np.nan

        with self.assertRaisesRegex(ValueError, 'X_train contains null feature values'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_infinite_feature_values(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        X_train.loc[10, 'feature_a'] = np.inf

        with self.assertRaisesRegex(ValueError, 'X_train contains non-finite feature values'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_float_label_dtype(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        y_test = y_test.astype(np.float64)

        with self.assertRaisesRegex(ValueError, 'y_test must be int64'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_non_binary_labels(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _valid_preprocessed_artifacts()
        )
        y_train.loc[10] = 2

        with self.assertRaisesRegex(ValueError, 'binary labels'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )

    def test_validate_preprocessed_data_fails_for_invalid_selected_features(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, _ = (
            _valid_preprocessed_artifacts()
        )
        selected_features = ['feature_a', 'feature_a']

        with self.assertRaisesRegex(ValueError, 'selected_features'):
            validation.validate_preprocessed_data(
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                selected_features,
            )


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


def _valid_preprocessed_artifacts(
    validation_rows: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Return structurally valid final preprocessing artifacts."""
    selected_features = ['feature_a', 'feature_b']
    X_train = pd.DataFrame(
        {
            'feature_a': np.array([0.0, 1.0, -1.0], dtype=np.float64),
            'feature_b': np.array([0.5, -0.5, 1.5], dtype=np.float64),
        },
        index=[10, 11, 12],
    )
    X_test = pd.DataFrame(
        {
            'feature_a': np.array([0.25, -0.25], dtype=np.float64),
            'feature_b': np.array([1.25, -1.25], dtype=np.float64),
        },
        index=[30, 31],
    )
    y_train = pd.Series(np.array([0, 1, 0], dtype=np.int64), index=X_train.index)
    y_test = pd.Series(np.array([1, 0], dtype=np.int64), index=X_test.index)

    if validation_rows == 0:
        X_validation = pd.DataFrame(
            columns=selected_features,
            index=pd.Index([], dtype=int),
            dtype=float,
        )
        y_validation = pd.Series(np.array([], dtype=np.int64), index=X_validation.index)
    else:
        X_validation = pd.DataFrame(
            {
                'feature_a': np.array([0.75, -0.75], dtype=np.float64),
                'feature_b': np.array([1.75, -1.75], dtype=np.float64),
            },
            index=[20, 21],
        )
        y_validation = pd.Series(
            np.resize([0, 1], validation_rows).astype(np.int64),
            index=X_validation.index[:validation_rows],
        )
        X_validation = X_validation.iloc[:validation_rows].copy()

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        selected_features,
    )


if __name__ == '__main__':
    unittest.main()
