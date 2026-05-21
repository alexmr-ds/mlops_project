"""Unit tests for preprocessing pipeline registration."""

import unittest

from mlops_project.pipeline_registry import register_pipelines
from mlops_project.pipelines.preprocessing import create_pipeline


class PipelineTests(unittest.TestCase):
    """Tests for pipeline assembly and registration."""

    def test_create_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        select_features_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'select_features_rfe_node'
        )
        apply_selected_features_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'apply_selected_features_node'
        )
        validate_preprocessed_data_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'validate_preprocessed_data_node'
        )

        self.assertEqual(len(node_names), 9)
        self.assertListEqual(
            node_names,
            [
                'validate_raw_data_node',
                'split_dataset_node',
                'engineer_features_node',
                'remove_training_outliers_node',
                'impute_features_node',
                'scale_features_node',
                'select_features_rfe_node',
                'apply_selected_features_node',
                'validate_preprocessed_data_node',
            ],
        )
        self.assertListEqual(
            list(select_features_node.outputs),
            ['X_train_candidate', 'selected_features_candidate', 'rfe_summary'],
        )
        self.assertListEqual(
            list(apply_selected_features_node.inputs),
            ['X_validation_scaled', 'X_test_scaled', 'selected_features_candidate'],
        )
        self.assertListEqual(
            list(apply_selected_features_node.outputs),
            ['X_validation_candidate', 'X_test_candidate'],
        )
        self.assertListEqual(
            list(validate_preprocessed_data_node.outputs),
            [
                'X_train',
                'X_validation',
                'X_test',
                'y_train',
                'y_validation',
                'y_test',
                'selected_features',
            ],
        )

    def test_register_pipelines_sets_default_pipeline(self) -> None:
        pipelines = register_pipelines()

        self.assertIn('preprocessing', pipelines)
        self.assertIn('modeling', pipelines)
        self.assertIn('__default__', pipelines)
        self.assertGreater(
            len(pipelines['__default__'].nodes),
            len(pipelines['preprocessing'].nodes),
        )
        self.assertGreater(
            len(pipelines['__default__'].nodes),
            len(pipelines['modeling'].nodes),
        )


if __name__ == '__main__':
    unittest.main()
