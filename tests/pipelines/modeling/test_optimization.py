"""Unit tests for RandomForest optimization helpers."""

import unittest
from unittest import mock

import optuna

from mlops_project.modeling import evaluation
from mlops_project.modeling import optimization


class RandomForestOptimizationTests(unittest.TestCase):
    """Tests for tuning configuration and parameter resolution."""

    def test_sample_optuna_parameter_supports_float_search_space(self) -> None:
        trial = optuna.trial.FixedTrial({'max_samples': 0.75})

        value = optimization.sample_optuna_parameter(
            trial,
            'max_samples',
            {'type': 'float', 'low': 0.6, 'high': 1.0},
        )

        self.assertEqual(value, 0.75)

    def test_sample_optuna_parameter_preserves_float_log_sampling(self) -> None:
        trial = mock.Mock()
        trial.suggest_float.return_value = 0.75

        value = optimization.sample_optuna_parameter(
            trial,
            'learning_rate',
            {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        )

        self.assertEqual(value, 0.75)
        trial.suggest_float.assert_called_once_with(
            'learning_rate',
            0.001,
            0.1,
            log=True,
        )

    def test_resolve_random_forest_parameters_drops_max_samples_without_bootstrap(
        self,
    ) -> None:
        resolved_parameters = optimization.resolve_random_forest_parameters(
            _modeling_parameters(),
            selected_params={
                'n_estimators': 20,
                'bootstrap': False,
                'max_samples': 0.8,
            },
        )

        self.assertFalse(resolved_parameters['bootstrap'])
        self.assertIsNone(resolved_parameters['max_samples'])
        self.assertEqual(resolved_parameters['random_state'], 73)
        self.assertEqual(resolved_parameters['n_jobs'], -1)

    def test_validate_random_forest_parameters_rejects_unfixed_random_state(self) -> None:
        with self.assertRaisesRegex(ValueError, 'random_state must remain fixed at 73'):
            optimization.validate_random_forest_parameters(
                {
                    'n_estimators': 20,
                    'random_state': 42,
                }
            )

    def test_primary_development_metric_is_shared_with_evaluation(self) -> None:
        self.assertEqual(
            optimization.PRIMARY_DEVELOPMENT_METRIC,
            evaluation.PRIMARY_DEVELOPMENT_METRIC,
        )


def _modeling_parameters() -> dict[str, object]:
    """Return minimal RandomForest modeling parameters."""
    return {
        'random_forest': {
            'n_estimators': 10,
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': None,
            'criterion': 'gini',
            'max_samples': None,
            'random_state': 73,
            'n_jobs': -1,
        },
        'random_forest_optimization': {
            'random_state': 73,
        },
    }


if __name__ == '__main__':
    unittest.main()
