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
            'random_state': 73,
            'shuffle': True,
            'stratify': True,
        }
        labels = [0, 1] * 20
        self.data = pd.DataFrame(
            {
                'ph': np.linspace(0.0, 1.0, 40),
                'Hardness': np.linspace(10.0, 20.0, 40),
                'Potability': labels,
            }
        )

    def test_split_dataset_returns_85_15_shapes(self) -> None:
        X_train, X_test, y_train, y_test = nodes.split_dataset(self.data, self.parameters)

        self.assertEqual(len(X_train), 34)
        self.assertEqual(len(X_test), 6)
        self.assertEqual(len(y_train), 34)
        self.assertEqual(len(y_test), 6)
        self.assertSetEqual(set(y_train.unique()), {0, 1})
        self.assertSetEqual(set(y_test.unique()), {0, 1})


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
        X_test = X_train.iloc[[0]].copy()

        engineered_train, engineered_test = nodes.engineer_features(X_train, X_test)

        self.assertAlmostEqual(
            engineered_train.loc[0, 'conductivity_solids_ratio'],
            0.5,
            places=6,
        )
        self.assertTrue(np.isfinite(engineered_train.loc[1, 'conductivity_solids_ratio']))
        self.assertEqual(engineered_train.loc[0, 'turbidity_trihalo_risk'], 1)
        self.assertEqual(engineered_train.loc[1, 'risk_score'], 3)
        self.assertEqual(engineered_train.loc[0, 'expanded_risk_score'], 3)
        self.assertEqual(engineered_train.loc[1, 'expanded_risk_score'], 4)
        self.assertEqual(engineered_train.loc[0, 'ph_safe_range'], 1)
        self.assertEqual(engineered_train.loc[1, 'high_chloramines'], 1)
        self.assertEqual(engineered_train.loc[0, 'high_hardness'], 0)
        self.assertIn('organic_trihalo_interaction', engineered_train.columns)
        self.assertEqual(engineered_test.shape[1], engineered_train.shape[1])

    def test_engineer_features_preserves_missing_values_for_downstream_imputation(
        self,
    ) -> None:
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

        engineered_train, _ = nodes.engineer_features(X_train, X_train.iloc[0:0])

        self.assertTrue(np.isnan(engineered_train.loc[0, 'chloramines_ph_interaction']))
        self.assertTrue(np.isnan(engineered_train.loc[0, 'disinfection_stress']))


if __name__ == '__main__':
    unittest.main()
