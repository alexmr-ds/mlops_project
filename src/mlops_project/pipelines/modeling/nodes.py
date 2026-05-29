"""Kedro node adapters for modeling workflows and reporting plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from mlops_project.modeling import evaluation as modeling_evaluation
from mlops_project.modeling import experiment_tracking as modeling_experiment_tracking
from mlops_project.modeling import optimization as modeling_optimization

CLASS_LABELS = modeling_evaluation.CLASS_LABELS


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
    return modeling_optimization.tune_random_forest_hyperparameters(
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
    modeling_evaluation.validate_feature_label_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    modeling_optimization.validate_random_forest_parameters(best_params)
    random_forest_parameters = modeling_optimization.resolve_random_forest_parameters(
        modeling_parameters,
        selected_params=best_params,
    )
    return modeling_evaluation.train_evaluate_model(
        model_name="random_forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_optimization.with_model_parameters(
            modeling_parameters,
            "random_forest",
            random_forest_parameters,
        ),
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
    run_name = parameters["mlflow"].get(
        "random_forest_run_name",
        modeling_experiment_tracking.RANDOM_FOREST_RUN_NAME,
    )
    return modeling_experiment_tracking.log_cv_model_to_mlflow(
        model=model,
        selected_features=selected_features,
        best_params=best_params,
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
        optuna_trials=optuna_trials,
        optuna_fold_metrics=optuna_fold_metrics,
        optuna_search_space=parameters["random_forest_optimization"]["search_space"],
        test_metrics=test_metrics,
        test_confusion_matrix=test_confusion_matrix,
        test_confusion_matrix_plot=test_confusion_matrix_plot,
        parameters=parameters,
        model_name="random_forest",
        run_name=str(run_name),
        artifact_model_name=modeling_experiment_tracking.RANDOM_FOREST_MODEL_ARTIFACT_NAME,
    )
