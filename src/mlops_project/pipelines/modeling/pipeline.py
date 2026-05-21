"""Kedro pipeline definition for baseline modeling."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the modeling pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.train_logistic_regression_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "selected_features",
                    "params:modeling",
                ],
                outputs="logistic_regression_model",
                name="train_logistic_regression_model_node",
            ),
            node(
                func=nodes.evaluate_validation_model,
                inputs=[
                    "logistic_regression_model",
                    "X_validation",
                    "y_validation",
                    "selected_features",
                ],
                outputs=["validation_metrics", "validation_confusion_matrix"],
                name="evaluate_validation_model_node",
            ),
            node(
                func=nodes.evaluate_test_model,
                inputs=[
                    "logistic_regression_model",
                    "X_test",
                    "y_test",
                    "selected_features",
                ],
                outputs=["test_metrics", "test_confusion_matrix"],
                name="evaluate_test_model_node",
            ),
            node(
                func=nodes.create_validation_confusion_matrix_plot,
                inputs="validation_confusion_matrix",
                outputs="validation_confusion_matrix_plot",
                name="create_validation_confusion_matrix_plot_node",
            ),
            node(
                func=nodes.create_test_confusion_matrix_plot,
                inputs="test_confusion_matrix",
                outputs="test_confusion_matrix_plot",
                name="create_test_confusion_matrix_plot_node",
            ),
            node(
                func=nodes.log_model_to_mlflow,
                inputs=[
                    "logistic_regression_model",
                    "selected_features",
                    "validation_metrics",
                    "test_metrics",
                    "validation_confusion_matrix",
                    "test_confusion_matrix",
                    "validation_confusion_matrix_plot",
                    "test_confusion_matrix_plot",
                    "params:modeling",
                ],
                outputs="mlflow_run_info",
                name="log_model_to_mlflow_node",
            ),
        ]
    )
