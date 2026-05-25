"""Kedro pipeline definitions for model training and evaluation."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the aggregate modeling pipeline."""
    del kwargs
    return create_logistic_regression_pipeline() + create_random_forest_pipeline()


def create_logistic_regression_pipeline(**kwargs: object) -> Pipeline:
    """Create the LogisticRegression modeling pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.cross_validate_and_train_logistic_regression,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:preprocessing",
                    "params:modeling",
                ],
                outputs=[
                    "logistic_regression_model",
                    "logistic_regression_cv_metrics",
                    "logistic_regression_cv_fold_metrics",
                    "logistic_regression_test_metrics",
                    "logistic_regression_test_confusion_matrix",
                    "logistic_regression_selected_features",
                ],
                name="cross_validate_and_train_logistic_regression_node",
            ),
            node(
                func=nodes.create_test_confusion_matrix_plot,
                inputs="logistic_regression_test_confusion_matrix",
                outputs="logistic_regression_test_confusion_matrix_plot",
                name="create_logistic_regression_test_confusion_matrix_plot_node",
            ),
            node(
                func=nodes.log_model_to_mlflow,
                inputs=[
                    "logistic_regression_model",
                    "logistic_regression_selected_features",
                    "logistic_regression_cv_metrics",
                    "logistic_regression_cv_fold_metrics",
                    "logistic_regression_test_metrics",
                    "logistic_regression_test_confusion_matrix",
                    "logistic_regression_test_confusion_matrix_plot",
                    "params:modeling",
                ],
                outputs="mlflow_run_info",
                name="log_model_to_mlflow_node",
            ),
        ]
    )


def create_random_forest_pipeline(**kwargs: object) -> Pipeline:
    """Create the RandomForestClassifier modeling pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.tune_random_forest_hyperparameters,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:preprocessing",
                    "params:modeling",
                ],
                outputs=[
                    "random_forest_best_params",
                    "random_forest_cv_metrics",
                    "random_forest_cv_fold_metrics",
                    "random_forest_optuna_trials",
                    "random_forest_optuna_fold_metrics",
                ],
                name="tune_random_forest_hyperparameters_node",
            ),
            node(
                func=nodes.train_evaluate_random_forest_with_best_params,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:preprocessing",
                    "params:modeling",
                    "random_forest_best_params",
                ],
                outputs=[
                    "random_forest_model",
                    "random_forest_test_metrics",
                    "random_forest_test_confusion_matrix",
                    "random_forest_selected_features",
                ],
                name="train_evaluate_random_forest_with_best_params_node",
            ),
            node(
                func=nodes.create_test_confusion_matrix_plot,
                inputs="random_forest_test_confusion_matrix",
                outputs="random_forest_test_confusion_matrix_plot",
                name="create_random_forest_test_confusion_matrix_plot_node",
            ),
            node(
                func=nodes.log_random_forest_to_mlflow,
                inputs=[
                    "random_forest_model",
                    "random_forest_selected_features",
                    "random_forest_best_params",
                    "random_forest_cv_metrics",
                    "random_forest_cv_fold_metrics",
                    "random_forest_optuna_trials",
                    "random_forest_optuna_fold_metrics",
                    "random_forest_test_metrics",
                    "random_forest_test_confusion_matrix",
                    "random_forest_test_confusion_matrix_plot",
                    "params:modeling",
                ],
                outputs="random_forest_mlflow_run_info",
                name="log_random_forest_to_mlflow_node",
            ),
        ]
    )
