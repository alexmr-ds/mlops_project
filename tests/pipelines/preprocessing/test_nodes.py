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
            'random_state': 73,
            'shuffle': True,
            'stratify': True,
            'zscore_threshold': 3.0,
            'rfe': {
                'enabled': True,
                'step': 1,
                'cv_folds': 5,
                'scoring': 'roc_auc',
                'n_jobs': -1,
                'logistic_max_iter': 5000,
                'random_state': 73,
            },
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


class FeatureEngineeringTests(unittest.TestCase):
    """Tests for deterministic feature engineering."""

    def test_engineer_features_adds_requested_columns(self) -> None:
        X_train = pd.DataFrame(
            {
                'ph': [7.0, 9.0],
                'Hardness': [200.0, 150.0],
                'Solids': [500.0, 0.0],
                'Conductivity': [250.0, 0.0],
                'Sulfate': [260.0, 401.0],
                'Chloramines': [3.0, 11.0],
                'Turbidity': [6.0, 4.0],
                'Organic_carbon': [10.0, 8.0],
                'Trihalomethanes': [90.0, 70.0],
            }
        )
        X_validation = X_train.iloc[0:0].copy()
        X_test = X_train.iloc[[0]].copy()

        engineered_train, engineered_validation, engineered_test = nodes.engineer_features(
            X_train, X_validation, X_test
        )

        self.assertAlmostEqual(engineered_train.loc[0, 'conductivity_solids_ratio'], 0.5, places=6)
        self.assertTrue(np.isfinite(engineered_train.loc[1, 'conductivity_solids_ratio']))
        self.assertEqual(engineered_train.loc[0, 'turbidity_trihalo_risk'], 1)
        self.assertEqual(engineered_train.loc[1, 'risk_score'], 3)
        self.assertEqual(engineered_train.loc[0, 'expanded_risk_score'], 3)
        self.assertEqual(engineered_train.loc[1, 'expanded_risk_score'], 4)
        self.assertEqual(engineered_train.loc[0, 'ph_safe_range'], 1)
        self.assertEqual(engineered_train.loc[1, 'high_chloramines'], 1)
        self.assertEqual(engineered_train.loc[0, 'high_hardness'], 0)
        self.assertIn('organic_trihalo_interaction', engineered_train.columns)
        self.assertListEqual(engineered_validation.columns.tolist(), X_validation.columns.tolist())
        self.assertEqual(engineered_test.shape[1], engineered_train.shape[1])

    def test_engineer_features_preserves_missing_values_for_downstream_imputation(self) -> None:
        X_train = pd.DataFrame(
            {
                'ph': [7.0],
                'Hardness': [200.0],
                'Solids': [500.0],
                'Conductivity': [250.0],
                'Sulfate': [260.0],
                'Chloramines': [np.nan],
                'Turbidity': [6.0],
                'Organic_carbon': [10.0],
                'Trihalomethanes': [90.0],
            }
        )

        engineered_train, _, _ = nodes.engineer_features(X_train, X_train.iloc[0:0], X_train.iloc[0:0])

        self.assertTrue(np.isnan(engineered_train.loc[0, 'chloramines_ph_interaction']))
        self.assertTrue(np.isnan(engineered_train.loc[0, 'disinfection_stress']))


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


class FeatureSelectionTests(unittest.TestCase):
    """Tests for RFECV selection and holdout masking."""

    def test_select_features_rfe_returns_selected_training_columns(self) -> None:
        row_count = 30
        signal = np.array([0] * 15 + [1] * 15)
        noise = np.tile([0.2, -0.2, 0.1], 10)
        weak_signal = signal * 0.3 + noise
        X_train = pd.DataFrame(
            {
                'signal': signal.astype(float),
                'weak_signal': weak_signal.astype(float),
                'noise': np.linspace(-1.0, 1.0, row_count),
            }
        )
        y_train = pd.Series(signal)
        parameters = {
            'rfe': {
                'enabled': True,
                'step': 1,
                'cv_folds': 3,
                'scoring': 'roc_auc',
                'n_jobs': -1,
                'logistic_max_iter': 5000,
                'random_state': 73,
            }
        }

        selected_train, selected_features, rfe_summary = nodes.select_features_rfe(
            X_train, y_train, parameters
        )

        self.assertGreaterEqual(len(selected_features), 1)
        self.assertListEqual(selected_train.columns.tolist(), selected_features)
        self.assertTrue(set(selected_features).issubset(X_train.columns))
        self.assertListEqual(
            rfe_summary.columns.tolist(),
            ['feature', 'selected', 'rank', 'feature_order'],
        )
        self.assertEqual(len(rfe_summary), X_train.shape[1])
        self.assertSetEqual(set(rfe_summary['feature']), set(X_train.columns))
        self.assertListEqual(
            rfe_summary.sort_values('feature_order')['feature'].tolist(),
            X_train.columns.tolist(),
        )
        self.assertListEqual(
            rfe_summary.loc[rfe_summary['selected'], 'feature'].tolist(),
            selected_features,
        )
        self.assertTrue((rfe_summary.loc[rfe_summary['selected'], 'rank'] == 1).all())

    def test_select_features_rfe_returns_complete_summary_when_disabled(self) -> None:
        X_train = pd.DataFrame(
            {
                'signal': [0.0, 1.0, 0.0, 1.0],
                'weak_signal': [0.2, 0.9, 0.1, 0.8],
                'noise': [-1.0, -0.5, 0.5, 1.0],
            }
        )
        y_train = pd.Series([0, 1, 0, 1])

        selected_train, selected_features, rfe_summary = nodes.select_features_rfe(
            X_train, y_train, {'rfe': {'enabled': False}}
        )

        self.assertListEqual(selected_train.columns.tolist(), X_train.columns.tolist())
        self.assertListEqual(selected_features, X_train.columns.tolist())
        self.assertTrue(rfe_summary['selected'].all())
        self.assertTrue((rfe_summary['rank'] == 1).all())
        self.assertListEqual(rfe_summary['feature'].tolist(), X_train.columns.tolist())
        self.assertListEqual(rfe_summary['feature_order'].tolist(), [1, 2, 3])

    def test_apply_selected_features_projects_validation_and_test(self) -> None:
        X_validation = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]}, index=[20])
        X_test = pd.DataFrame({'a': [4.0], 'b': [5.0], 'c': [6.0]}, index=[30])
        selected_features = ['c', 'a']

        selected_validation, selected_test = nodes.apply_selected_features(
            X_validation, X_test, selected_features
        )

        self.assertListEqual(selected_validation.columns.tolist(), selected_features)
        self.assertListEqual(selected_test.columns.tolist(), selected_features)
        self.assertEqual(selected_validation.loc[20, 'c'], 3.0)
        self.assertEqual(selected_test.loc[30, 'a'], 4.0)

    def test_apply_selected_features_keeps_empty_validation_split(self) -> None:
        X_validation = pd.DataFrame(columns=['a', 'b'])
        X_test = pd.DataFrame({'a': [1.0], 'b': [2.0]})

        selected_validation, _ = nodes.apply_selected_features(
            X_validation, X_test, ['b']
        )

        self.assertTrue(selected_validation.empty)
        self.assertListEqual(selected_validation.columns.tolist(), ['b'])


if __name__ == '__main__':
    unittest.main()
