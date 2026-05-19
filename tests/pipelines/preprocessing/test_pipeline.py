"""Unit tests for preprocessing pipeline registration."""

import unittest

from mlops_project.pipeline_registry import register_pipelines
from mlops_project.pipelines.preprocessing import create_pipeline


class PipelineTests(unittest.TestCase):
    """Tests for pipeline assembly and registration."""

    def test_create_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]

        self.assertEqual(len(node_names), 7)
        self.assertListEqual(
            node_names,
            [
                'split_dataset_node',
                'engineer_features_node',
                'remove_training_outliers_node',
                'impute_features_node',
                'scale_features_node',
                'select_features_rfe_node',
                'apply_selected_features_node',
            ],
        )

    def test_register_pipelines_sets_default_pipeline(self) -> None:
        pipelines = register_pipelines()

        self.assertIn('preprocessing', pipelines)
        self.assertIn('__default__', pipelines)
        self.assertEqual(pipelines['preprocessing'].describe(), pipelines['__default__'].describe())


if __name__ == '__main__':
    unittest.main()
