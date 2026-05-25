"""Unit tests for modeling pipeline assembly."""

from pathlib import Path
import tempfile
import unittest

from kedro.config import OmegaConfigLoader
import pandas as pd

from mlops_project import datasets
from mlops_project.pipelines.modeling import create_logistic_regression_pipeline
from mlops_project.pipelines.modeling import create_pipeline
from mlops_project.pipelines.modeling import create_random_forest_pipeline

LOGISTIC_REGRESSION_NODE_NAMES = [
    'cross_validate_and_train_logistic_regression_node',
    'create_logistic_regression_test_confusion_matrix_plot_node',
    'log_model_to_mlflow_node',
]
RANDOM_FOREST_NODE_NAMES = [
    'cross_validate_and_train_random_forest_node',
    'create_random_forest_test_confusion_matrix_plot_node',
    'log_random_forest_to_mlflow_node',
]


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
        random_forest_train_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'cross_validate_and_train_random_forest_node'
        )
        random_forest_log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_random_forest_to_mlflow_node'
        )

        self.assertCountEqual(node_names, RANDOM_FOREST_NODE_NAMES)
        self.assertListEqual(
            list(random_forest_train_node.outputs),
            [
                'random_forest_model',
                'random_forest_cv_metrics',
                'random_forest_cv_fold_metrics',
                'random_forest_test_metrics',
                'random_forest_test_confusion_matrix',
                'random_forest_selected_features',
            ],
        )
        self.assertListEqual(
            list(random_forest_log_node.outputs),
            ['random_forest_mlflow_run_info'],
        )

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
            LOGISTIC_REGRESSION_NODE_NAMES + RANDOM_FOREST_NODE_NAMES,
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


def _catalog_config() -> dict[str, object]:
    """Return the project catalog configuration."""
    project_root = Path(__file__).resolve().parents[3]
    config_loader = OmegaConfigLoader(conf_source=str(project_root / 'conf'))
    return config_loader['catalog']


if __name__ == '__main__':
    unittest.main()
