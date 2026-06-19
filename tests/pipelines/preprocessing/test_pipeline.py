"""Unit tests for preprocessing pipeline registration."""

import unittest

from mlops_project.pipeline_registry import register_pipelines
from mlops_project.pipelines.preprocessing import create_pipeline


class PipelineTests(unittest.TestCase):
    """Tests for pipeline assembly and registration."""

    def test_create_pipeline_exposes_expected_nodes(self) -> None:
        pipeline = create_pipeline()
        node_names = [pipeline_node.name for pipeline_node in pipeline.nodes]
        validate_modeling_input_data_node = next(
            pipeline_node
            for pipeline_node in pipeline.nodes
            if pipeline_node.name == 'validate_modeling_input_data_node'
        )

        self.assertEqual(len(node_names), 4)
        self.assertListEqual(
            node_names,
            [
                'validate_raw_data_node',
                'split_dataset_node',
                'engineer_features_node',
                'validate_modeling_input_data_node',
            ],
        )
        self.assertListEqual(
            list(validate_modeling_input_data_node.outputs),
            [
                'X_train',
                'X_test',
                'y_train',
                'y_test',
            ],
        )

    def test_register_pipelines_sets_default_pipeline(self) -> None:
        pipelines = register_pipelines()

        self.assertIn('preprocessing', pipelines)
        self.assertIn('modeling_logistic_regression', pipelines)
        self.assertIn('modeling_random_forest', pipelines)
        self.assertIn('modeling_extra_trees', pipelines)
        self.assertIn('modeling_hist_gradient_boosting', pipelines)
        self.assertIn('modeling_xgboost', pipelines)
        self.assertIn('modeling', pipelines)
        self.assertIn('__default__', pipelines)
        self.assertIn('data_drift', pipelines)
        self.assertEqual(
            len(pipelines['modeling'].nodes),
            len(pipelines['modeling_logistic_regression'].nodes)
            + len(pipelines['modeling_random_forest'].nodes)
            + len(pipelines['modeling_extra_trees'].nodes)
            + len(pipelines['modeling_hist_gradient_boosting'].nodes)
            + len(pipelines['modeling_xgboost'].nodes)
            + 2,  # model_comparison node + SHAP node
        )
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
