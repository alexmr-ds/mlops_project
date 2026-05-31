"""Unit tests for modeling pipeline assembly."""

from pathlib import Path
import tempfile
import unittest

from kedro.config import OmegaConfigLoader
import pandas as pd

from mlops_project import datasets
from mlops_project.pipelines.modeling import create_extra_trees_pipeline
from mlops_project.pipelines.modeling import create_hist_gradient_boosting_pipeline
from mlops_project.pipelines.modeling import create_logistic_regression_pipeline
from mlops_project.pipelines.modeling import create_pipeline
from mlops_project.pipelines.modeling import create_random_forest_pipeline
from mlops_project.pipelines.modeling import create_xgboost_pipeline

LOGISTIC_REGRESSION_NODE_NAMES = [
    'cross_validate_and_train_logistic_regression_node',
    'create_logistic_regression_test_confusion_matrix_plot_node',
    'log_model_to_mlflow_node',
]
RANDOM_FOREST_NODE_NAMES = [
    'tune_random_forest_hyperparameters_node',
    'train_evaluate_random_forest_with_best_params_node',
    'create_random_forest_test_confusion_matrix_plot_node',
    'log_random_forest_to_mlflow_node',
]
EXTRA_TREES_NODE_NAMES = [
    'tune_extra_trees_hyperparameters_node',
    'train_evaluate_extra_trees_with_best_params_node',
    'create_extra_trees_test_confusion_matrix_plot_node',
    'log_extra_trees_to_mlflow_node',
]
HIST_GRADIENT_BOOSTING_NODE_NAMES = [
    'tune_hist_gradient_boosting_hyperparameters_node',
    'train_evaluate_hist_gradient_boosting_with_best_params_node',
    'create_hist_gradient_boosting_test_confusion_matrix_plot_node',
    'log_hist_gradient_boosting_to_mlflow_node',
]
XGBOOST_NODE_NAMES = [
    'tune_xgboost_hyperparameters_node',
    'train_evaluate_xgboost_with_best_params_node',
    'create_xgboost_test_confusion_matrix_plot_node',
    'log_xgboost_to_mlflow_node',
]
COMPARISON_NODE_NAMES = ['build_model_comparison_node']


class ModelingPipelineTests(unittest.TestCase):
    """Tests for modeling pipeline node wiring."""

    def test_create_logistic_regression_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_logistic_regression_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        train_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'cross_validate_and_train_logistic_regression_node'
        )
        log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_model_to_mlflow_node'
        )

        self.assertCountEqual(node_names, LOGISTIC_REGRESSION_NODE_NAMES)
        self.assertListEqual(
            list(train_node.outputs),
            [
                'logistic_regression_model',
                'logistic_regression_cv_metrics',
                'logistic_regression_cv_fold_metrics',
                'logistic_regression_test_metrics',
                'logistic_regression_test_confusion_matrix',
                'logistic_regression_selected_features',
            ],
        )
        self.assertListEqual(
            list(log_node.outputs),
            ['mlflow_run_info'],
        )

    def test_create_random_forest_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_random_forest_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        random_forest_tuning_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'tune_random_forest_hyperparameters_node'
        )
        random_forest_train_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'train_evaluate_random_forest_with_best_params_node'
        )
        random_forest_log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_random_forest_to_mlflow_node'
        )

        self.assertCountEqual(node_names, RANDOM_FOREST_NODE_NAMES)
        self.assertListEqual(
            list(random_forest_tuning_node.outputs),
            [
                'random_forest_best_params',
                'random_forest_cv_metrics',
                'random_forest_cv_fold_metrics',
                'random_forest_optuna_trials',
                'random_forest_optuna_fold_metrics',
            ],
        )
        self.assertListEqual(
            list(random_forest_train_node.outputs),
            [
                'random_forest_model',
                'random_forest_test_metrics',
                'random_forest_test_confusion_matrix',
                'random_forest_selected_features',
            ],
        )
        self.assertListEqual(
            list(random_forest_log_node.outputs),
            ['random_forest_mlflow_run_info'],
        )
        self.assertIn('random_forest_best_params', random_forest_log_node.inputs)
        self.assertIn('random_forest_optuna_trials', random_forest_log_node.inputs)
        self.assertIn('random_forest_optuna_fold_metrics', random_forest_log_node.inputs)

    def test_create_new_tree_pipelines_expose_expected_nodes(self) -> None:
        cases = [
            (
                'extra_trees',
                create_extra_trees_pipeline(),
                EXTRA_TREES_NODE_NAMES,
            ),
            (
                'hist_gradient_boosting',
                create_hist_gradient_boosting_pipeline(),
                HIST_GRADIENT_BOOSTING_NODE_NAMES,
            ),
            ('xgboost', create_xgboost_pipeline(), XGBOOST_NODE_NAMES),
        ]

        for model_name, model_pipeline, expected_node_names in cases:
            with self.subTest(model_name=model_name):
                node_names = [pipeline_node.name for pipeline_node in model_pipeline.nodes]
                tuning_node = next(
                    pipeline_node
                    for pipeline_node in model_pipeline.nodes
                    if pipeline_node.name == f'tune_{model_name}_hyperparameters_node'
                )
                train_node = next(
                    pipeline_node
                    for pipeline_node in model_pipeline.nodes
                    if pipeline_node.name
                    == f'train_evaluate_{model_name}_with_best_params_node'
                )
                log_node = next(
                    pipeline_node
                    for pipeline_node in model_pipeline.nodes
                    if pipeline_node.name == f'log_{model_name}_to_mlflow_node'
                )

                self.assertCountEqual(node_names, expected_node_names)
                self.assertListEqual(
                    list(tuning_node.outputs),
                    [
                        f'{model_name}_best_params',
                        f'{model_name}_cv_metrics',
                        f'{model_name}_cv_fold_metrics',
                        f'{model_name}_optuna_trials',
                        f'{model_name}_optuna_fold_metrics',
                    ],
                )
                self.assertListEqual(
                    list(train_node.outputs),
                    [
                        f'{model_name}_model',
                        f'{model_name}_test_metrics',
                        f'{model_name}_test_confusion_matrix',
                        f'{model_name}_selected_features',
                    ],
                )
                self.assertListEqual(
                    list(log_node.outputs),
                    [f'{model_name}_mlflow_run_info'],
                )
                self.assertIn(f'{model_name}_best_params', log_node.inputs)
                self.assertIn(f'{model_name}_optuna_trials', log_node.inputs)
                self.assertIn(f'{model_name}_optuna_fold_metrics', log_node.inputs)

    def test_create_pipeline_exposes_all_modeling_nodes(self) -> None:
        pipeline = create_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_model_to_mlflow_node'
        )
        random_forest_log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_random_forest_to_mlflow_node'
        )

        self.assertCountEqual(
            node_names,
            (
                LOGISTIC_REGRESSION_NODE_NAMES
                + RANDOM_FOREST_NODE_NAMES
                + EXTRA_TREES_NODE_NAMES
                + HIST_GRADIENT_BOOSTING_NODE_NAMES
                + XGBOOST_NODE_NAMES
                + COMPARISON_NODE_NAMES
            ),
        )
        self.assertListEqual(
            list(log_node.outputs),
            ['mlflow_run_info'],
        )
        self.assertListEqual(
            list(random_forest_log_node.outputs),
            ['random_forest_mlflow_run_info'],
        )

    def test_confusion_matrix_catalog_preserves_index_on_load(self) -> None:
        catalog = _catalog_config()
        matrix = pd.DataFrame(
            [[2, 1], [1, 2]],
            index=[0, 1],
            columns=[0, 1],
        )

        for dataset_name in [
            'logistic_regression_test_confusion_matrix',
            'random_forest_test_confusion_matrix',
            'extra_trees_test_confusion_matrix',
            'hist_gradient_boosting_test_confusion_matrix',
            'xgboost_test_confusion_matrix',
        ]:
            dataset_config = catalog[dataset_name]
            self.assertEqual(dataset_config['save_args'], {'index': True})
            self.assertEqual(dataset_config['load_args'], {'index_col': 0})

            with tempfile.TemporaryDirectory() as temporary_directory:
                dataset = datasets.PandasCSVDataset(
                    filepath=str(Path(temporary_directory) / f'{dataset_name}.csv'),
                    load_args=dataset_config['load_args'],
                    save_args=dataset_config['save_args'],
                )
                dataset.save(matrix)

                reloaded_matrix = dataset.load()

            self.assertEqual(reloaded_matrix.shape, (2, 2))

    def test_random_forest_optimization_outputs_are_cataloged(self) -> None:
        catalog = _catalog_config()

        self.assertIn('random_forest_best_params', catalog)
        self.assertIn('random_forest_optuna_trials', catalog)
        self.assertIn('random_forest_optuna_fold_metrics', catalog)

    def test_tuned_model_outputs_are_cataloged(self) -> None:
        catalog = _catalog_config()

        for model_name in [
            'random_forest',
            'extra_trees',
            'hist_gradient_boosting',
            'xgboost',
        ]:
            with self.subTest(model_name=model_name):
                self.assertIn(f'{model_name}_model', catalog)
                self.assertIn(f'{model_name}_selected_features', catalog)
                self.assertIn(f'{model_name}_best_params', catalog)
                self.assertIn(f'{model_name}_cv_metrics', catalog)
                self.assertIn(f'{model_name}_cv_fold_metrics', catalog)
                self.assertIn(f'{model_name}_test_metrics', catalog)
                self.assertIn(f'{model_name}_optuna_trials', catalog)
                self.assertIn(f'{model_name}_optuna_fold_metrics', catalog)
                self.assertIn(f'{model_name}_mlflow_run_info', catalog)
        self.assertIn('model_comparison', catalog)

    def test_project_parameters_satisfy_tuned_model_pipeline_inputs(self) -> None:
        parameters = _parameters_config()
        self.assertIn('modeling', parameters)

        for model_name, model_pipeline in [
            ('random_forest', create_random_forest_pipeline()),
            ('extra_trees', create_extra_trees_pipeline()),
            ('hist_gradient_boosting', create_hist_gradient_boosting_pipeline()),
            ('xgboost', create_xgboost_pipeline()),
        ]:
            with self.subTest(model_name=model_name):
                parameter_inputs = {
                    pipeline_input.removeprefix('params:')
                    for pipeline_input in model_pipeline.inputs()
                    if pipeline_input.startswith('params:')
                }
                self.assertTrue(parameter_inputs.issubset(parameters))
                self.assertIn(model_name, parameters['modeling'])
                self.assertIn(f'{model_name}_optimization', parameters['modeling'])
                self.assertIn('mlflow', parameters['modeling'])


def _catalog_config() -> dict[str, object]:
    """Return the project catalog configuration."""
    project_root = Path(__file__).resolve().parents[3]
    config_loader = OmegaConfigLoader(conf_source=str(project_root / 'conf'))
    return config_loader['catalog']


def _parameters_config() -> dict[str, object]:
    """Return the project parameters configuration."""
    project_root = Path(__file__).resolve().parents[3]
    config_loader = OmegaConfigLoader(conf_source=str(project_root / 'conf'))
    return config_loader['parameters']


if __name__ == '__main__':
    unittest.main()
