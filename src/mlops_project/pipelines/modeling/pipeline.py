"""Kedro pipeline definitions for model training and evaluation."""

from collections.abc import Callable
from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

TunedNodeFunction = Callable[
    [Any, Any, Any, dict[str, Any]],
    tuple[dict[str, Any], Any, Any, Any, Any],
]


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the aggregate modeling pipeline."""
    del kwargs
    return (
        create_logistic_regression_pipeline()
        + create_random_forest_pipeline()
        + create_extra_trees_pipeline()
        + create_hist_gradient_boosting_pipeline()
        + create_xgboost_pipeline()
        + create_model_comparison_pipeline()
        + create_random_forest_shap_pipeline()
    )


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
    return create_tuned_model_pipeline(
        model_name="random_forest",
        tune_function=nodes.tune_random_forest_hyperparameters,
        train_function=nodes.train_evaluate_random_forest_with_best_params,
        log_function=nodes.log_random_forest_to_mlflow,
    )


def create_extra_trees_pipeline(**kwargs: object) -> Pipeline:
    """Create the ExtraTreesClassifier modeling pipeline."""
    del kwargs
    return create_tuned_model_pipeline(
        model_name="extra_trees",
        tune_function=nodes.tune_extra_trees_hyperparameters,
        train_function=nodes.train_evaluate_extra_trees_with_best_params,
        log_function=nodes.log_extra_trees_to_mlflow,
    )


def create_hist_gradient_boosting_pipeline(**kwargs: object) -> Pipeline:
    """Create the HistGradientBoostingClassifier modeling pipeline."""
    del kwargs
    return create_tuned_model_pipeline(
        model_name="hist_gradient_boosting",
        tune_function=nodes.tune_hist_gradient_boosting_hyperparameters,
        train_function=nodes.train_evaluate_hist_gradient_boosting_with_best_params,
        log_function=nodes.log_hist_gradient_boosting_to_mlflow,
    )


def create_xgboost_pipeline(**kwargs: object) -> Pipeline:
    """Create the XGBoost modeling pipeline."""
    del kwargs
    return create_tuned_model_pipeline(
        model_name="xgboost",
        tune_function=nodes.tune_xgboost_hyperparameters,
        train_function=nodes.train_evaluate_xgboost_with_best_params,
        log_function=nodes.log_xgboost_to_mlflow,
    )


def create_tuned_model_pipeline(
    *,
    model_name: str,
    tune_function: TunedNodeFunction,
    train_function: Callable[..., tuple[Any, Any, Any, list[str]]],
    log_function: Callable[..., Any],
) -> Pipeline:
    """Create one tuned model-family pipeline."""
    return pipeline(
        [
            node(
                func=tune_function,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:preprocessing",
                    "params:modeling",
                ],
                outputs=[
                    f"{model_name}_best_params",
                    f"{model_name}_cv_metrics",
                    f"{model_name}_cv_fold_metrics",
                    f"{model_name}_optuna_trials",
                    f"{model_name}_optuna_fold_metrics",
                ],
                name=f"tune_{model_name}_hyperparameters_node",
            ),
            node(
                func=train_function,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:preprocessing",
                    "params:modeling",
                    f"{model_name}_best_params",
                ],
                outputs=[
                    f"{model_name}_model",
                    f"{model_name}_test_metrics",
                    f"{model_name}_test_confusion_matrix",
                    f"{model_name}_selected_features",
                ],
                name=f"train_evaluate_{model_name}_with_best_params_node",
            ),
            node(
                func=nodes.create_test_confusion_matrix_plot,
                inputs=f"{model_name}_test_confusion_matrix",
                outputs=f"{model_name}_test_confusion_matrix_plot",
                name=f"create_{model_name}_test_confusion_matrix_plot_node",
            ),
            node(
                func=log_function,
                inputs=[
                    f"{model_name}_model",
                    f"{model_name}_selected_features",
                    f"{model_name}_best_params",
                    f"{model_name}_cv_metrics",
                    f"{model_name}_cv_fold_metrics",
                    f"{model_name}_optuna_trials",
                    f"{model_name}_optuna_fold_metrics",
                    f"{model_name}_test_metrics",
                    f"{model_name}_test_confusion_matrix",
                    f"{model_name}_test_confusion_matrix_plot",
                    "params:modeling",
                ],
                outputs=f"{model_name}_mlflow_run_info",
                name=f"log_{model_name}_to_mlflow_node",
            ),
        ]
    )


def create_random_forest_shap_pipeline(**kwargs: object) -> Pipeline:
    """Create the SHAP explainability pipeline for the best-performing Random Forest."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.compute_random_forest_shap_values,
                inputs=["random_forest_model", "X_test"],
                outputs=["random_forest_shap_summary", "random_forest_shap_summary_plot"],
                name="compute_random_forest_shap_values_node",
            ),
        ]
    )


def create_model_comparison_pipeline(**kwargs: object) -> Pipeline:
    """Create the aggregate model comparison reporting pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.build_model_comparison,
                inputs=[
                    "logistic_regression_cv_metrics",
                    "logistic_regression_test_metrics",
                    "random_forest_cv_metrics",
                    "random_forest_test_metrics",
                    "extra_trees_cv_metrics",
                    "extra_trees_test_metrics",
                    "hist_gradient_boosting_cv_metrics",
                    "hist_gradient_boosting_test_metrics",
                    "xgboost_cv_metrics",
                    "xgboost_test_metrics",
                ],
                outputs="model_comparison",
                name="build_model_comparison_node",
            )
        ]
    )
