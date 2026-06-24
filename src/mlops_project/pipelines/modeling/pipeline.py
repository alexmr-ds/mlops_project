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
    """Create the aggregate, reproducible modeling pipeline.

    This default pipeline refits every tuned model from its persisted, committed
    best parameters (no Optuna), so repeated runs reproduce the same artifacts.
    Hyperparameter search lives in the separate, opt-in ``tuning`` pipeline.
    """
    del kwargs
    return (
        create_logistic_regression_pipeline()
        + create_random_forest_pipeline()
        + create_extra_trees_pipeline()
        + create_hist_gradient_boosting_pipeline()
        + create_xgboost_pipeline()
        + create_model_comparison_pipeline()
        + create_extra_trees_shap_pipeline()
    )


def create_tuning_pipeline(**kwargs: object) -> Pipeline:
    """Create the aggregate, opt-in Optuna tuning pipeline for all tree-based models.

    This is intentionally NOT part of the default run. Optuna's TPE search is
    sequential and sensitive to floating-point differences, so it is not
    bit-reproducible across machines; running it rewrites the committed
    ``*_best_params`` and Optuna trial logs that the reproducible ``modeling``
    pipeline then refits from.
    """
    del kwargs
    return (
        create_tuning_random_forest_pipeline()
        + create_tuning_extra_trees_pipeline()
        + create_tuning_hist_gradient_boosting_pipeline()
        + create_tuning_xgboost_pipeline()
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
    """Create the reproducible RandomForest modeling pipeline (refit from locked params)."""
    del kwargs
    return create_refit_model_pipeline(
        model_name="random_forest",
        cv_function=nodes.cross_validate_random_forest_with_best_params,
        train_function=nodes.train_evaluate_random_forest_with_best_params,
        log_function=nodes.log_random_forest_to_mlflow,
    )


def create_extra_trees_pipeline(**kwargs: object) -> Pipeline:
    """Create the reproducible ExtraTrees modeling pipeline (refit from locked params)."""
    del kwargs
    return create_refit_model_pipeline(
        model_name="extra_trees",
        cv_function=nodes.cross_validate_extra_trees_with_best_params,
        train_function=nodes.train_evaluate_extra_trees_with_best_params,
        log_function=nodes.log_extra_trees_to_mlflow,
    )


def create_hist_gradient_boosting_pipeline(**kwargs: object) -> Pipeline:
    """Create the reproducible HistGradientBoosting modeling pipeline (refit from locked params)."""
    del kwargs
    return create_refit_model_pipeline(
        model_name="hist_gradient_boosting",
        cv_function=nodes.cross_validate_hist_gradient_boosting_with_best_params,
        train_function=nodes.train_evaluate_hist_gradient_boosting_with_best_params,
        log_function=nodes.log_hist_gradient_boosting_to_mlflow,
    )


def create_xgboost_pipeline(**kwargs: object) -> Pipeline:
    """Create the reproducible XGBoost modeling pipeline (refit from locked params)."""
    del kwargs
    return create_refit_model_pipeline(
        model_name="xgboost",
        cv_function=nodes.cross_validate_xgboost_with_best_params,
        train_function=nodes.train_evaluate_xgboost_with_best_params,
        log_function=nodes.log_xgboost_to_mlflow,
    )


def create_tuning_random_forest_pipeline(**kwargs: object) -> Pipeline:
    """Create the opt-in Optuna tuning pipeline for RandomForest."""
    del kwargs
    return create_tuning_model_pipeline("random_forest", nodes.tune_random_forest_hyperparameters)


def create_tuning_extra_trees_pipeline(**kwargs: object) -> Pipeline:
    """Create the opt-in Optuna tuning pipeline for ExtraTrees."""
    del kwargs
    return create_tuning_model_pipeline("extra_trees", nodes.tune_extra_trees_hyperparameters)


def create_tuning_hist_gradient_boosting_pipeline(**kwargs: object) -> Pipeline:
    """Create the opt-in Optuna tuning pipeline for HistGradientBoosting."""
    del kwargs
    return create_tuning_model_pipeline(
        "hist_gradient_boosting", nodes.tune_hist_gradient_boosting_hyperparameters
    )


def create_tuning_xgboost_pipeline(**kwargs: object) -> Pipeline:
    """Create the opt-in Optuna tuning pipeline for XGBoost."""
    del kwargs
    return create_tuning_model_pipeline("xgboost", nodes.tune_xgboost_hyperparameters)


def create_refit_model_pipeline(
    *,
    model_name: str,
    cv_function: Callable[..., tuple[Any, Any]],
    train_function: Callable[..., tuple[Any, Any, Any, list[str]]],
    log_function: Callable[..., Any],
) -> Pipeline:
    """Create one reproducible model-family pipeline that refits from locked params.

    The committed ``{model}_best_params`` and Optuna trial logs are read as free
    inputs (produced by the opt-in tuning pipeline); no Optuna runs here, so the
    cross-validation, refit, and final evaluation are deterministic.
    """
    return pipeline(
        [
            node(
                func=cv_function,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:preprocessing",
                    "params:modeling",
                    f"{model_name}_best_params",
                ],
                outputs=[
                    f"{model_name}_cv_metrics",
                    f"{model_name}_cv_fold_metrics",
                ],
                name=f"cross_validate_{model_name}_with_best_params_node",
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


def create_tuning_model_pipeline(
    model_name: str,
    tune_function: TunedNodeFunction,
) -> Pipeline:
    """Create one opt-in Optuna tuning pipeline that rewrites locked params and logs."""
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
        ]
    )


def create_extra_trees_shap_pipeline(**kwargs: object) -> Pipeline:
    """Create the SHAP explainability pipeline for the best-performing Extra Trees model."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.compute_extra_trees_shap_values,
                inputs=["extra_trees_model", "X_test"],
                outputs=["extra_trees_shap_summary", "extra_trees_shap_summary_plot"],
                name="compute_extra_trees_shap_values_node",
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
