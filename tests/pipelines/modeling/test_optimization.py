"""Unit tests for model optimization helpers."""

import unittest
from unittest import mock

import optuna

from mlops_project.modeling import evaluation
from mlops_project.modeling import optimization


class ModelOptimizationTests(unittest.TestCase):
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

    def test_resolve_extra_trees_parameters_drops_max_samples_without_bootstrap(
        self,
    ) -> None:
        resolved_parameters = optimization.resolve_model_parameters(
            _modeling_parameters(),
            'extra_trees',
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

    def test_model_optimization_config_uses_model_specific_key(self) -> None:
        config = optimization.model_optimization_config(
            'xgboost',
            _modeling_parameters(),
        )

        self.assertEqual(config['n_trials'], 2)
        self.assertIn('learning_rate', config['search_space'])

    def test_validate_model_parameters_rejects_unfixed_random_state(self) -> None:
        with self.assertRaisesRegex(ValueError, 'random_state must remain fixed at 73'):
            optimization.validate_model_parameters(
                'xgboost',
                {
                    'n_estimators': 20,
                    'random_state': 42,
                },
            )

    def test_validate_model_parameters_rejects_unfixed_n_jobs(self) -> None:
        with self.assertRaisesRegex(ValueError, 'n_jobs must remain fixed at -1'):
            optimization.validate_model_parameters(
                'extra_trees',
                {
                    'n_estimators': 20,
                    'random_state': 73,
                    'n_jobs': 1,
                },
            )

    def test_primary_development_metric_is_shared_with_evaluation(self) -> None:
        self.assertEqual(
            optimization.PRIMARY_DEVELOPMENT_METRIC,
            evaluation.PRIMARY_DEVELOPMENT_METRIC,
        )


def _modeling_parameters() -> dict[str, object]:
    """Return minimal tuned modeling parameters."""
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
            'n_trials': 2,
            'objective_metric': 'f1',
            'search_space': {'n_estimators': {'type': 'categorical', 'choices': [10]}},
        },
        'extra_trees': {
            'n_estimators': 10,
            'bootstrap': True,
            'max_samples': None,
            'random_state': 73,
            'n_jobs': -1,
        },
        'extra_trees_optimization': {
            'random_state': 73,
            'n_trials': 2,
            'objective_metric': 'f1',
            'search_space': {'n_estimators': {'type': 'categorical', 'choices': [10]}},
        },
        'xgboost': {
            'n_estimators': 10,
            'random_state': 73,
            'n_jobs': -1,
        },
        'xgboost_optimization': {
            'random_state': 73,
            'n_trials': 2,
            'objective_metric': 'f1',
            'search_space': {
                'learning_rate': {'type': 'categorical', 'choices': [0.1]}
            },
        },
    }


if __name__ == '__main__':
    unittest.main()
