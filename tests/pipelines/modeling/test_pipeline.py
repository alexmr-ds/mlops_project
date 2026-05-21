"""Unit tests for modeling pipeline assembly."""

from pathlib import Path
import tempfile
import unittest

from kedro.config import OmegaConfigLoader
import pandas as pd

from mlops_project import datasets
from mlops_project.pipelines.modeling import create_pipeline


class ModelingPipelineTests(unittest.TestCase):
    """Tests for modeling pipeline node wiring."""

    def test_create_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        log_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'log_model_to_mlflow_node'
        )

        self.assertCountEqual(
            node_names,
            [
                'train_logistic_regression_model_node',
                'evaluate_validation_model_node',
                'evaluate_test_model_node',
                'create_validation_confusion_matrix_plot_node',
                'create_test_confusion_matrix_plot_node',
                'log_model_to_mlflow_node',
            ],
        )
        self.assertListEqual(
            list(log_node.outputs),
            ['mlflow_run_info'],
        )

    def test_confusion_matrix_catalog_preserves_index_on_load(self) -> None:
        catalog = _catalog_config()
        matrix = pd.DataFrame(
            [[2, 1], [1, 2]],
            index=['actual_0', 'actual_1'],
            columns=['predicted_0', 'predicted_1'],
        )

        for dataset_name in [
            'validation_confusion_matrix',
            'test_confusion_matrix',
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
            self.assertListEqual(
                reloaded_matrix.columns.tolist(),
                ['predicted_0', 'predicted_1'],
            )
            self.assertListEqual(
                reloaded_matrix.index.tolist(),
                ['actual_0', 'actual_1'],
            )


def _catalog_config() -> dict[str, object]:
    """Return the project catalog configuration."""
    project_root = Path(__file__).resolve().parents[3]
    config_loader = OmegaConfigLoader(conf_source=str(project_root / 'conf'))
    return config_loader['catalog']


if __name__ == '__main__':
    unittest.main()
