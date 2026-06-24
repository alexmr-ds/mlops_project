"""Kedro node adapters for modeling workflows and reporting plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from mlops_project.modeling import evaluation as modeling_evaluation
from mlops_project.modeling import experiment_tracking as modeling_experiment_tracking
from mlops_project.modeling import explainability as modeling_explainability
from mlops_project.modeling import optimization as modeling_optimization

CLASS_LABELS = modeling_evaluation.CLASS_LABELS
TUNED_MODEL_DEFAULT_RUN_NAMES = {
    "random_forest": modeling_experiment_tracking.RANDOM_FOREST_RUN_NAME,
    "extra_trees": "extra_trees_nonlinear_probe",
    "hist_gradient_boosting": "hist_gradient_boosting_nonlinear_probe",
    "xgboost": "xgboost_nonlinear_probe",
}


def cross_validate_and_train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cross-validate, train, and test a logistic regression model."""
    return modeling_evaluation.cross_validate_and_train_model(
        model_name="logistic_regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )


def cross_validate_and_train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cross-validate, train, and test a random forest classifier."""
    return modeling_evaluation.cross_validate_and_train_model(
        model_name="random_forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )


def tune_random_forest_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune random forest hyperparameters with training-set-only CV."""
    return tune_model_hyperparameters(
        "random_forest",
        X_train,
        y_train,
        preprocessing_parameters,
        modeling_parameters,
    )


def tune_extra_trees_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune extra trees hyperparameters with training-set-only CV."""
    return tune_model_hyperparameters(
        "extra_trees",
        X_train,
        y_train,
        preprocessing_parameters,
        modeling_parameters,
    )


def tune_hist_gradient_boosting_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune histogram gradient boosting hyperparameters with training-set-only CV."""
    return tune_model_hyperparameters(
        "hist_gradient_boosting",
        X_train,
        y_train,
        preprocessing_parameters,
        modeling_parameters,
    )


def tune_xgboost_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune XGBoost hyperparameters with training-set-only CV."""
    return tune_model_hyperparameters(
        "xgboost",
        X_train,
        y_train,
        preprocessing_parameters,
        modeling_parameters,
    )


def tune_model_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune one model family with training-set-only CV."""
    return modeling_optimization.tune_model_hyperparameters(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )


def train_evaluate_random_forest_with_best_params(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit the selected random forest and evaluate the final holdout split."""
    return train_evaluate_model_with_best_params(
        "random_forest",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_parameters,
        modeling_parameters,
        best_params,
    )


def train_evaluate_extra_trees_with_best_params(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit the selected extra trees model and evaluate the final holdout split."""
    return train_evaluate_model_with_best_params(
        "extra_trees",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_parameters,
        modeling_parameters,
        best_params,
    )


def train_evaluate_hist_gradient_boosting_with_best_params(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit the selected histogram gradient boosting model and test it."""
    return train_evaluate_model_with_best_params(
        "hist_gradient_boosting",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_parameters,
        modeling_parameters,
        best_params,
    )


def train_evaluate_xgboost_with_best_params(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit the selected XGBoost model and evaluate the final holdout split."""
    return train_evaluate_model_with_best_params(
        "xgboost",
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_parameters,
        modeling_parameters,
        best_params,
    )


def train_evaluate_model_with_best_params(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit one selected tuned model and evaluate the final holdout split."""
    modeling_evaluation.validate_feature_label_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    modeling_optimization.validate_model_parameters(model_name, best_params)
    model_parameters = modeling_optimization.resolve_model_parameters(
        modeling_parameters,
        model_name,
        selected_params=best_params,
    )
    return modeling_evaluation.train_evaluate_model(
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_optimization.with_model_parameters(
            modeling_parameters,
            model_name,
            model_parameters,
        ),
    )


def cross_validate_with_best_params(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate one tuned model family from persisted best parameters.

    Used by the reproducible default modeling pipeline, which refits from locked
    parameters instead of re-running Optuna. The resulting CV metrics match the
    best-trial CV metrics recorded by the opt-in tuning pipeline because both run
    the same training-set cross-validation with the same parameters.
    """
    modeling_optimization.validate_model_parameters(model_name, best_params)
    model_parameters = modeling_optimization.resolve_model_parameters(
        modeling_parameters,
        model_name,
        selected_params=best_params,
    )
    result = modeling_evaluation.cross_validate_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_optimization.with_model_parameters(
            modeling_parameters,
            model_name,
            model_parameters,
        ),
    )
    return result.cv_metrics, result.cv_fold_metrics


def cross_validate_random_forest_with_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the random forest from persisted best parameters."""
    return cross_validate_with_best_params(
        "random_forest", X_train, y_train, preprocessing_parameters, modeling_parameters, best_params
    )


def cross_validate_extra_trees_with_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the extra trees model from persisted best parameters."""
    return cross_validate_with_best_params(
        "extra_trees", X_train, y_train, preprocessing_parameters, modeling_parameters, best_params
    )


def cross_validate_hist_gradient_boosting_with_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the histogram gradient boosting model from persisted best parameters."""
    return cross_validate_with_best_params(
        "hist_gradient_boosting", X_train, y_train, preprocessing_parameters, modeling_parameters, best_params
    )


def cross_validate_xgboost_with_best_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
    best_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the XGBoost model from persisted best parameters."""
    return cross_validate_with_best_params(
        "xgboost", X_train, y_train, preprocessing_parameters, modeling_parameters, best_params
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted model on the final holdout test split."""
    return modeling_evaluation.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        selected_features=selected_features,
    )


def create_test_confusion_matrix_plot(confusion_matrix_frame: pd.DataFrame) -> Figure:
    """Create a matplotlib plot for the final test confusion matrix."""
    modeling_evaluation.validate_confusion_matrix_frame(confusion_matrix_frame)

    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(confusion_matrix_frame.values, cmap="Blues")

    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Actual Label")
    axis.set_xticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS)
    axis.set_yticks(range(len(CLASS_LABELS)), labels=CLASS_LABELS)

    for row_index, row_label in enumerate(confusion_matrix_frame.index):
        for column_index, column_label in enumerate(confusion_matrix_frame.columns):
            axis.text(
                column_index,
                row_index,
                str(confusion_matrix_frame.loc[row_label, column_label]),
                ha="center",
                va="center",
                color="black",
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def create_confusion_matrix_plot(confusion_matrix_frame: pd.DataFrame) -> Figure:
    """Create a matplotlib plot for a confusion matrix."""
    return create_test_confusion_matrix_plot(confusion_matrix_frame)


def log_model_to_mlflow(
    model: Any,
    selected_features: list[str],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log logistic regression CV and final test outputs to MLflow."""
    return modeling_experiment_tracking.log_cv_model_to_mlflow(
        model=model,
        selected_features=selected_features,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
        model_name="logistic_regression",
        run_name=parameters["mlflow"]["run_name"],
        artifact_model_name=(
            modeling_experiment_tracking.LOGISTIC_REGRESSION_MODEL_ARTIFACT_NAME
        ),
    )


def log_random_forest_to_mlflow(
    model: Any,
    selected_features: list[str],
    best_params: dict[str, Any],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log random forest CV and final test outputs to MLflow."""
    return log_tuned_model_to_mlflow(
        model_name="random_forest",
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
    )


def log_extra_trees_to_mlflow(
    model: Any,
    selected_features: list[str],
    best_params: dict[str, Any],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log extra trees CV and final test outputs to MLflow."""
    return log_tuned_model_to_mlflow(
        model_name="extra_trees",
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
    )


def log_hist_gradient_boosting_to_mlflow(
    model: Any,
    selected_features: list[str],
    best_params: dict[str, Any],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log histogram gradient boosting CV and final test outputs to MLflow."""
    return log_tuned_model_to_mlflow(
        model_name="hist_gradient_boosting",
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
    )


def log_xgboost_to_mlflow(
    model: Any,
    selected_features: list[str],
    best_params: dict[str, Any],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log XGBoost CV and final test outputs to MLflow."""
    return log_tuned_model_to_mlflow(
        model_name="xgboost",
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
    )


def log_tuned_model_to_mlflow(
    *,
    model_name: str,
    model: Any,
    selected_features: list[str],
    best_params: dict[str, Any],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    optuna_trials: pd.DataFrame,
    optuna_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log a tuned model family's CV and final test outputs to MLflow."""
    optimization_config_name = modeling_optimization.optimization_config_name(model_name)
    optimization_config = parameters.get(optimization_config_name, {})
    run_name = parameters["mlflow"].get(
        f"{model_name}_run_name",
        TUNED_MODEL_DEFAULT_RUN_NAMES[model_name],
    )
    return modeling_experiment_tracking.log_cv_model_to_mlflow(
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        optuna_search_space=optimization_config.get("search_space"),
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
        model_name=model_name,
        run_name=str(run_name),
        artifact_model_name=f"{model_name}_model",
    )


def build_model_comparison(
    logistic_regression_cv_metrics: pd.DataFrame,
    logistic_regression_test_metrics: pd.DataFrame,
    random_forest_cv_metrics: pd.DataFrame,
    random_forest_test_metrics: pd.DataFrame,
    extra_trees_cv_metrics: pd.DataFrame,
    extra_trees_test_metrics: pd.DataFrame,
    hist_gradient_boosting_cv_metrics: pd.DataFrame,
    hist_gradient_boosting_test_metrics: pd.DataFrame,
    xgboost_cv_metrics: pd.DataFrame,
    xgboost_test_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a model comparison report ranked by development CV F1."""
    rows = [
        build_model_comparison_row(
            "logistic_regression",
            logistic_regression_cv_metrics,
            logistic_regression_test_metrics,
        ),
        build_model_comparison_row(
            "random_forest",
            random_forest_cv_metrics,
            random_forest_test_metrics,
        ),
        build_model_comparison_row(
            "extra_trees",
            extra_trees_cv_metrics,
            extra_trees_test_metrics,
        ),
        build_model_comparison_row(
            "hist_gradient_boosting",
            hist_gradient_boosting_cv_metrics,
            hist_gradient_boosting_test_metrics,
        ),
        build_model_comparison_row(
            "xgboost",
            xgboost_cv_metrics,
            xgboost_test_metrics,
        ),
    ]
    comparison = pd.DataFrame(rows).sort_values(
        by=modeling_evaluation.PRIMARY_DEVELOPMENT_METRIC,
        ascending=False,
    )
    comparison.insert(0, "development_rank", range(1, len(comparison) + 1))
    return comparison.reset_index(drop=True)


def compute_extra_trees_shap_values(
    model: Any,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, Figure]:
    """Compute SHAP values for the fitted Extra Trees champion and produce a summary plot.

    Returns the per-feature summary DataFrame and the bar-chart Figure so both
    can be persisted independently through the Kedro catalog.
    """
    shap_values, shap_summary = modeling_explainability.compute_shap_values(
        model_bundle=model,
        X_test=X_test,
    )
    shap_plot = modeling_explainability.create_shap_summary_plot(
        shap_values, shap_summary, model_label="Extra Trees"
    )
    return shap_summary, shap_plot


def build_model_comparison_row(
    model_name: str,
    cv_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Build one model comparison row from CV and final holdout metrics."""
    modeling_experiment_tracking.validate_cv_metrics_frame(cv_metrics)
    modeling_experiment_tracking.validate_test_metrics_frame(test_metrics)
    row: dict[str, Any] = {"model_name": model_name}
    row.update(cv_metrics.iloc[0].to_dict())
    row.update(
        {
            f"test_{metric_name}": test_metrics.iloc[0][metric_name]
            for metric_name in modeling_evaluation.METRIC_NAMES
        }
    )
    return row
