"""Unit tests for preprocessing nodes."""

import unittest

import numpy as np
import pandas as pd

from mlops_project.pipelines.preprocessing import nodes


class SplitDatasetTests(unittest.TestCase):
    """Tests for dataset splitting behavior."""

    def setUp(self) -> None:
        self.parameters = {
            'target_column': 'Potability',
            'test_size': 0.15,
            'validation_size': 0.15,
            'random_state': 42,
            'shuffle': True,
            'stratify': True,
            'zscore_threshold': 3.0,
        }
        labels = [0, 1] * 20
        self.data = pd.DataFrame(
            {
                'ph': np.linspace(0.0, 1.0, 40),
                'Hardness': np.linspace(10.0, 20.0, 40),
                'Potability': labels,
            }
        )

    def test_split_dataset_returns_70_15_15_shapes(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test = nodes.split_dataset(
            self.data, self.parameters
        )

        self.assertEqual(len(X_train), 28)
        self.assertEqual(len(X_validation), 6)
        self.assertEqual(len(X_test), 6)
        self.assertEqual(len(y_train), 28)
        self.assertEqual(len(y_validation), 6)
        self.assertEqual(len(y_test), 6)
        self.assertSetEqual(set(y_train.unique()), {0, 1})
        self.assertSetEqual(set(y_validation.unique()), {0, 1})
        self.assertSetEqual(set(y_test.unique()), {0, 1})

    def test_split_dataset_can_disable_validation_outputs(self) -> None:
        parameters = dict(self.parameters)
        parameters['validation_size'] = 0.0

        X_train, X_validation, X_test, y_train, y_validation, y_test = nodes.split_dataset(
            self.data, parameters
        )

        self.assertEqual(len(X_train), 34)
        self.assertTrue(X_validation.empty)
        self.assertEqual(len(X_test), 6)
        self.assertTrue(y_validation.empty)
        self.assertEqual(len(y_train), 34)
        self.assertEqual(len(y_test), 6)


class OutlierRemovalTests(unittest.TestCase):
    """Tests for training outlier removal."""

    def test_remove_outliers_filters_rows_above_threshold(self) -> None:
        X_train = pd.DataFrame(
            {
                'ph': [7.0, 7.1, 6.9, 50.0],
                'Hardness': [200.0, 201.0, 199.0, 700.0],
            },
            index=[10, 11, 12, 13],
        )
        y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
        parameters = {'feature_columns': ['ph', 'Hardness'], 'zscore_threshold': 1.4}

        filtered_X, filtered_y = nodes.remove_outliers(X_train, y_train, parameters)

        self.assertListEqual(filtered_X.index.tolist(), [10, 11, 12])
        self.assertListEqual(filtered_y.index.tolist(), [10, 11, 12])

    def test_remove_outliers_keeps_nan_rows_with_nan_policy_omit(self) -> None:
        X_train = pd.DataFrame(
            {
                'ph': [7.0, np.nan, 7.2],
                'Hardness': [200.0, 201.0, 199.0],
            }
        )
        y_train = pd.Series([0, 1, 0])
        parameters = {'zscore_threshold': 3.0}

        filtered_X, filtered_y = nodes.remove_outliers(X_train, y_train, parameters)

        self.assertEqual(len(filtered_X), 3)
        self.assertEqual(len(filtered_y), 3)


class ScalingTests(unittest.TestCase):
    """Tests for standard scaling."""

    def test_scale_features_preserves_columns_and_centers_train_data(self) -> None:
        X_train = pd.DataFrame({'ph': [1.0, 2.0, 3.0], 'Hardness': [10.0, 20.0, 30.0]})
        X_validation = pd.DataFrame({'ph': [4.0], 'Hardness': [40.0]})
        X_test = pd.DataFrame({'ph': [5.0], 'Hardness': [50.0]})

        scaled_train, scaled_validation, scaled_test = nodes.scale_features(
            X_train, X_validation, X_test
        )

        self.assertListEqual(scaled_train.columns.tolist(), ['ph', 'Hardness'])
        self.assertTrue(np.allclose(scaled_train.mean().to_numpy(), [0.0, 0.0]))
        self.assertTrue(np.allclose(scaled_train.std(ddof=0).to_numpy(), [1.0, 1.0]))
        self.assertEqual(scaled_validation.shape, (1, 2))
        self.assertEqual(scaled_test.shape, (1, 2))

    def test_scale_features_keeps_empty_validation_split(self) -> None:
        X_train = pd.DataFrame({'ph': [1.0, 2.0], 'Hardness': [10.0, 20.0]})
        X_validation = pd.DataFrame(columns=['ph', 'Hardness'])
        X_test = pd.DataFrame({'ph': [3.0], 'Hardness': [30.0]})

        _, scaled_validation, _ = nodes.scale_features(X_train, X_validation, X_test)

        self.assertTrue(scaled_validation.empty)
        self.assertListEqual(scaled_validation.columns.tolist(), ['ph', 'Hardness'])


class ImputationTests(unittest.TestCase):
    """Tests for mean imputation."""

    def test_impute_features_uses_training_means_for_all_splits(self) -> None:
        X_train = pd.DataFrame(
            {'ph': [1.0, np.nan, 3.0], 'Hardness': [10.0, 20.0, 30.0]},
            index=[10, 11, 12],
        )
        X_validation = pd.DataFrame(
            {'ph': [np.nan], 'Hardness': [40.0]},
            index=[20],
        )
        X_test = pd.DataFrame(
            {'ph': [np.nan], 'Hardness': [50.0]},
            index=[30],
        )

        imputed_train, imputed_validation, imputed_test = nodes.impute_features(
            X_train, X_validation, X_test
        )

        self.assertEqual(imputed_train.loc[11, 'ph'], 2.0)
        self.assertEqual(imputed_validation.loc[20, 'ph'], 2.0)
        self.assertEqual(imputed_test.loc[30, 'ph'], 2.0)
        self.assertListEqual(imputed_train.columns.tolist(), ['ph', 'Hardness'])
        self.assertListEqual(imputed_train.index.tolist(), [10, 11, 12])

    def test_impute_features_keeps_empty_validation_split(self) -> None:
        X_train = pd.DataFrame({'ph': [1.0, np.nan], 'Hardness': [10.0, 20.0]})
        X_validation = pd.DataFrame(columns=['ph', 'Hardness'])
        X_test = pd.DataFrame({'ph': [3.0], 'Hardness': [30.0]})

        _, imputed_validation, _ = nodes.impute_features(X_train, X_validation, X_test)

        self.assertTrue(imputed_validation.empty)
        self.assertListEqual(imputed_validation.columns.tolist(), ['ph', 'Hardness'])


if __name__ == '__main__':
    unittest.main()
