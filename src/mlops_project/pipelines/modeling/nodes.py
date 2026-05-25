"""Node functions for cross-validated model training and evaluation."""

from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from mlops_project.modeling import model_bundle as modeling_model_bundle
from mlops_project.modeling import preprocessing as modeling_preprocessing


CLASS_LABELS = [0, 1]
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "f1_weighted", "roc_auc")
PRIMARY_DEVELOPMENT_METRIC = "cv_mean_f1"
LOGISTIC_REGRESSION_MODEL_ARTIFACT_NAME = "logistic_regression_model"
RANDOM_FOREST_MODEL_ARTIFACT_NAME = "random_forest_model"
RANDOM_FOREST_RUN_NAME = "random_forest_nonlinear_probe"
OPTUNA_TRIAL_PARAM_PREFIX = "param_"


class _ModelBundlePyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc adapter for persisted model bundles."""

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        del context
        return self._model_bundle.predict(model_input)

    def load_context(self, context: Any) -> None:
        """Load the bundled model artifact before serving predictions."""
        model_bundle_path = Path(context.artifacts["model_bundle"])
        with model_bundle_path.open("rb") as model_bundle_file:
            self._model_bundle = pickle.load(model_bundle_file)


@dataclass(frozen=True)
class FittedModelingStack:
    """Container for a fitted fold-local preprocessing and model stack."""

    model: modeling_model_bundle.ModelBundle
    selected_features: list[str]
    evaluation_features: pd.DataFrame
    filtered_training_row_count: int


@dataclass(frozen=True)
class CrossValidationResult:
    """Container for cross-validation summary and fold metrics."""

    cv_metrics: pd.DataFrame
    cv_fold_metrics: pd.DataFrame


def cross_validate_and_train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cross-validate, train, and test a logistic regression model."""
    return _cross_validate_and_train_model(
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
    return _cross_validate_and_train_model(
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
    _validate_training_artifacts(X_train, y_train)
    optimization_config = _random_forest_optimization_config(modeling_parameters)
    trial_fold_frames: list[pd.DataFrame] = []

    if not bool(optimization_config.get("enabled", True)):
        best_params = _resolve_random_forest_parameters(modeling_parameters)
        cv_result = _cross_validate_model(
            model_name="random_forest",
            X_train=X_train,
            y_train=y_train,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=_with_model_parameters(
                modeling_parameters,
                "random_forest",
                best_params,
            ),
        )
        trial_frame, trial_fold_frame = _build_disabled_optimization_artifacts(
            best_params,
            cv_result,
        )
        return (
            best_params,
            cv_result.cv_metrics,
            cv_result.cv_fold_metrics,
            trial_frame,
            trial_fold_frame,
        )

    sampler = _build_optuna_sampler(optimization_config)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        trial_params = _sample_random_forest_parameters(trial, optimization_config)
        model_params = _resolve_random_forest_parameters(
            modeling_parameters,
            selected_params=trial_params,
        )
        cv_result = _cross_validate_model(
            model_name="random_forest",
            X_train=X_train,
            y_train=y_train,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=_with_model_parameters(
                modeling_parameters,
                "random_forest",
                model_params,
            ),
        )
        objective_value = float(cv_result.cv_metrics.loc[0, PRIMARY_DEVELOPMENT_METRIC])
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("cv_metrics", cv_result.cv_metrics.iloc[0].to_dict())
        trial_fold_frames.append(
            _add_trial_context_to_fold_metrics(
                cv_result.cv_fold_metrics,
                trial.number,
                model_params,
            )
        )
        return objective_value

    study.optimize(
        objective,
        n_trials=int(optimization_config["n_trials"]),
        catch=(),
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["model_params"])
    best_cv_metrics = pd.DataFrame([best_trial.user_attrs["cv_metrics"]])
    trial_frame = _build_optuna_trial_summary_frame(study)
    trial_fold_frame = pd.concat(trial_fold_frames, ignore_index=True)
    best_fold_metrics = trial_fold_frame.loc[
        trial_fold_frame["trial_number"] == best_trial.number,
        _cv_fold_output_columns(),
    ].reset_index(drop=True)
    return (
        best_params,
        best_cv_metrics,
        best_fold_metrics,
        trial_frame,
        trial_fold_frame,
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
    _validate_feature_label_artifacts(X_train, X_test, y_train, y_test)
    _validate_random_forest_parameters(best_params)
    random_forest_parameters = _resolve_random_forest_parameters(
        modeling_parameters,
        selected_params=best_params,
    )
    final_stack = _fit_modeling_stack(
        model_name="random_forest",
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=_with_model_parameters(
            modeling_parameters,
            "random_forest",
            random_forest_parameters,
        ),
    )
    test_metrics, test_confusion_matrix = evaluate_model(
        model=final_stack.model,
        X_test=final_stack.evaluation_features,
        y_test=y_test,
        selected_features=final_stack.selected_features,
    )
    return (
        final_stack.model,
        test_metrics,
        test_confusion_matrix,
        final_stack.selected_features,
    )


def create_test_confusion_matrix_plot(confusion_matrix_frame: pd.DataFrame) -> Figure:
    """Create a matplotlib plot for the final test confusion matrix."""
    _validate_confusion_matrix_frame(confusion_matrix_frame)

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
    return _log_cv_model_to_mlflow(
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
        artifact_model_name=LOGISTIC_REGRESSION_MODEL_ARTIFACT_NAME,
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
    return _log_cv_model_to_mlflow(
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
        model_name="random_forest",
        run_name=RANDOM_FOREST_RUN_NAME,
        artifact_model_name=RANDOM_FOREST_MODEL_ARTIFACT_NAME,
    )


def _cross_validate_and_train_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    _validate_feature_label_artifacts(X_train, X_test, y_train, y_test)
    cv_result = _cross_validate_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )

    final_stack = _fit_modeling_stack(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_evaluation=X_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )
    test_metrics, test_confusion_matrix = evaluate_model(
        model=final_stack.model,
        X_test=final_stack.evaluation_features,
        y_test=y_test,
        selected_features=final_stack.selected_features,
    )

    return (
        final_stack.model,
        cv_result.cv_metrics,
        cv_result.cv_fold_metrics,
        test_metrics,
        test_confusion_matrix,
        final_stack.selected_features,
    )


def _cross_validate_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> CrossValidationResult:
    _validate_training_artifacts(X_train, y_train)
    cross_validation_config = modeling_parameters.get("cross_validation", {})
    splitter = _build_cross_validation_splitter(y_train, cross_validation_config)

    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        fold_X_train = X_train.iloc[train_positions]
        fold_y_train = y_train.iloc[train_positions]
        fold_X_validation = X_train.iloc[validation_positions]
        fold_y_validation = y_train.iloc[validation_positions]

        fitted_stack = _fit_modeling_stack(
            model_name=model_name,
            X_train=fold_X_train,
            y_train=fold_y_train,
            X_evaluation=fold_X_validation,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=modeling_parameters,
        )
        fold_metrics = _calculate_metrics(
            y_true=fold_y_validation,
            predictions=fitted_stack.model.predict(fitted_stack.evaluation_features),
            probabilities=_predict_positive_class_probability(
                fitted_stack.model,
                fitted_stack.evaluation_features,
            ),
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "train_row_count": len(fold_X_train),
                "filtered_train_row_count": fitted_stack.filtered_training_row_count,
                "validation_row_count": len(fold_X_validation),
                "selected_feature_count": len(fitted_stack.selected_features),
                **fold_metrics,
            }
        )

    cv_fold_metrics = pd.DataFrame(fold_rows)
    cv_metrics = _summarize_cv_metrics(cv_fold_metrics)
    return CrossValidationResult(
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted model on the final holdout test split."""
    _validate_test_artifacts(model, X_test, y_test, selected_features)

    evaluation_features = X_test
    if selected_features is not None and not isinstance(
        model, modeling_model_bundle.ModelBundle
    ):
        evaluation_features = X_test.loc[:, selected_features]

    predictions = model.predict(evaluation_features)
    probabilities = _predict_positive_class_probability(model, evaluation_features)
    metrics_frame = pd.DataFrame([_calculate_metrics(y_test, predictions, probabilities)])
    confusion_matrix_frame = _build_confusion_matrix_frame(y_test, predictions)
    return metrics_frame, confusion_matrix_frame


def _fit_modeling_stack(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> FittedModelingStack:
    _validate_fold_artifacts(X_train, X_evaluation, y_train)

    preprocessor = modeling_preprocessing.ModelPreprocessor(
        preprocessing_parameters,
    )
    selected_train, filtered_y_train = preprocessor.fit_resample(X_train, y_train)

    estimator = _build_model(model_name, modeling_parameters)
    estimator.fit(selected_train, filtered_y_train)
    model = modeling_model_bundle.ModelBundle(
        model_name=model_name,
        preprocessor=preprocessor,
        estimator=estimator,
    )

    return FittedModelingStack(
        model=model,
        selected_features=preprocessor.selected_features,
        evaluation_features=X_evaluation,
        filtered_training_row_count=preprocessor.filtered_training_row_count,
    )


def _build_cross_validation_splitter(
    y_train: pd.Series,
    cross_validation_config: dict[str, Any],
) -> StratifiedKFold:
    n_splits = int(cross_validation_config.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("Cross-validation n_splits must be at least 2.")

    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Cross-validation requires both target classes in the training split.")

    min_class_count = int(class_counts.min())
    if n_splits > min_class_count:
        raise ValueError(
            "Cross-validation n_splits cannot exceed the smallest class count. "
            f"Received n_splits={n_splits} and min_class_count={min_class_count}."
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(cross_validation_config.get("shuffle", True)),
        random_state=int(cross_validation_config.get("random_state", 73)),
    )


def _build_model(model_name: str, modeling_parameters: dict[str, Any]) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(**modeling_parameters["logistic_regression"])
    if model_name == "random_forest":
        return RandomForestClassifier(**modeling_parameters["random_forest"])
    raise ValueError(f"Unsupported model name: {model_name}")


def _random_forest_optimization_config(
    modeling_parameters: dict[str, Any],
) -> dict[str, Any]:
    config = modeling_parameters.get("random_forest_optimization", {})
    if not isinstance(config, dict):
        raise ValueError("random_forest_optimization must be a mapping.")
    if "n_trials" not in config:
        raise ValueError("random_forest_optimization.n_trials is required.")
    if int(config["n_trials"]) < 1:
        raise ValueError("random_forest_optimization.n_trials must be at least 1.")
    if str(config.get("objective_metric", "f1")) != "f1":
        raise ValueError("Random forest optimization currently supports only f1.")
    if bool(config.get("enabled", True)):
        search_space = config.get("search_space")
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError("random_forest_optimization.search_space must not be empty.")
    return config


def _build_optuna_sampler(
    optimization_config: dict[str, Any],
) -> optuna.samplers.BaseSampler:
    sampler_name = str(optimization_config.get("sampler", "tpe"))
    random_state = int(optimization_config.get("random_state", 73))
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=random_state)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


def _sample_random_forest_parameters(
    trial: optuna.Trial,
    optimization_config: dict[str, Any],
) -> dict[str, Any]:
    search_space = optimization_config["search_space"]
    return {
        parameter_name: _sample_optuna_parameter(trial, parameter_name, parameter_config)
        for parameter_name, parameter_config in search_space.items()
    }


def _sample_optuna_parameter(
    trial: optuna.Trial,
    parameter_name: str,
    parameter_config: dict[str, Any],
) -> Any:
    parameter_type = str(parameter_config["type"])
    if parameter_type == "int":
        return trial.suggest_int(
            parameter_name,
            int(parameter_config["low"]),
            int(parameter_config["high"]),
            step=int(parameter_config.get("step", 1)),
        )
    if parameter_type == "categorical":
        choices = parameter_config.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{parameter_name} categorical choices must not be empty.")
        return trial.suggest_categorical(parameter_name, choices)
    raise ValueError(f"Unsupported search space type for {parameter_name}: {parameter_type}")


def _resolve_random_forest_parameters(
    modeling_parameters: dict[str, Any],
    selected_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = dict(modeling_parameters["random_forest"])
    if selected_params:
        parameters.update(selected_params)
    optimization_config = modeling_parameters.get("random_forest_optimization", {})
    parameters["random_state"] = int(
        optimization_config.get("random_state", parameters.get("random_state", 73))
    )
    parameters["n_jobs"] = int(parameters.get("n_jobs", -1))
    _validate_random_forest_parameters(parameters)
    return parameters


def _validate_random_forest_parameters(parameters: dict[str, Any]) -> None:
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("random_forest parameters must be a non-empty mapping.")
    if int(parameters.get("random_state", 73)) != 73:
        raise ValueError("random_forest random_state must remain fixed at 73.")
    if "n_estimators" not in parameters:
        raise ValueError("random_forest parameters must include n_estimators.")


def _with_model_parameters(
    modeling_parameters: dict[str, Any],
    model_name: str,
    model_parameters: dict[str, Any],
) -> dict[str, Any]:
    updated_parameters = dict(modeling_parameters)
    updated_parameters[model_name] = dict(model_parameters)
    return updated_parameters


def _build_optuna_trial_summary_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_trial_number = study.best_trial.number
    for trial in study.trials:
        model_params = dict(trial.user_attrs["model_params"])
        cv_metrics = dict(trial.user_attrs["cv_metrics"])
        rows.append(
            {
                "trial_number": trial.number,
                "objective_value": float(trial.value),
                "is_best": trial.number == best_trial_number,
                **_prefix_parameters(model_params),
                **cv_metrics,
            }
        )
    return pd.DataFrame(rows)


def _build_disabled_optimization_artifacts(
    best_params: dict[str, Any],
    cv_result: CrossValidationResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_frame = pd.DataFrame(
        [
            {
                "trial_number": 0,
                "objective_value": float(
                    cv_result.cv_metrics.loc[0, PRIMARY_DEVELOPMENT_METRIC]
                ),
                "is_best": True,
                **_prefix_parameters(best_params),
                **cv_result.cv_metrics.iloc[0].to_dict(),
            }
        ]
    )
    trial_fold_frame = _add_trial_context_to_fold_metrics(
        cv_result.cv_fold_metrics,
        trial_number=0,
        parameters=best_params,
    )
    return trial_frame, trial_fold_frame


def _add_trial_context_to_fold_metrics(
    cv_fold_metrics: pd.DataFrame,
    trial_number: int,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    trial_fold_metrics = cv_fold_metrics.copy()
    trial_fold_metrics.insert(0, "trial_number", trial_number)
    for parameter_name, parameter_value in _prefix_parameters(parameters).items():
        trial_fold_metrics[parameter_name] = parameter_value
    return trial_fold_metrics


def _prefix_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{OPTUNA_TRIAL_PARAM_PREFIX}{parameter_name}": parameter_value
        for parameter_name, parameter_value in sorted(parameters.items())
    }


def _cv_fold_output_columns() -> list[str]:
    return [
        "fold",
        "train_row_count",
        "filtered_train_row_count",
        "validation_row_count",
        "selected_feature_count",
        *METRIC_NAMES,
    ]


def _calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "roc_auc": _safe_roc_auc_score(y_true, probabilities),
    }


def _safe_roc_auc_score(y_true: pd.Series, probabilities: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def _predict_positive_class_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    positive_class_index = list(model.classes_).index(1)
    return probabilities[:, positive_class_index]


def _summarize_cv_metrics(cv_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        metric_values = cv_fold_metrics[metric_name]
        summary[f"cv_mean_{metric_name}"] = float(metric_values.mean())
        summary[f"cv_std_{metric_name}"] = float(metric_values.std(ddof=0))
    return pd.DataFrame([summary])


def _build_confusion_matrix_frame(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
    return pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)


def _log_cv_model_to_mlflow(
    *,
    model: Any,
    selected_features: list[str],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
    model_name: str,
    run_name: str,
    artifact_model_name: str,
    best_params: dict[str, Any] | None = None,
    optuna_trials: pd.DataFrame | None = None,
    optuna_fold_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _validate_mlflow_inputs(
        model,
        selected_features,
        cv_metrics,
        cv_fold_metrics,
        test_metrics,
        test_confusion_matrix,
        test_confusion_matrix_plot,
        best_params,
        optuna_trials,
        optuna_fold_metrics,
    )

    mlflow_config = parameters["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    experiment = mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "primary_development_metric": PRIMARY_DEVELOPMENT_METRIC,
                "selected_feature_count": len(selected_features),
            }
        )
        mlflow.log_params(model.get_params())
        if best_params is not None:
            mlflow.log_params(
                {
                    f"best_{parameter_name}": _mlflow_param_value(parameter_value)
                    for parameter_name, parameter_value in best_params.items()
                }
            )
        if optuna_trials is not None:
            _log_optuna_trial_metadata(optuna_trials)
        _log_metrics_frame(cv_metrics)
        _log_test_metrics_frame(test_metrics)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            selected_features_path = temporary_path / "selected_features.json"
            cv_metrics_path = temporary_path / "cv_metrics.csv"
            cv_fold_metrics_path = temporary_path / "cv_fold_metrics.csv"
            test_confusion_matrix_path = temporary_path / "test_confusion_matrix.csv"
            test_confusion_matrix_plot_path = temporary_path / "test_confusion_matrix.png"
            model_bundle_path = temporary_path / "model_bundle.pkl"
            best_params_path = temporary_path / "best_params.json"
            optuna_trials_path = temporary_path / "optuna_trials.csv"
            optuna_fold_metrics_path = temporary_path / "optuna_fold_metrics.csv"

            selected_features_path.write_text(
                json.dumps(selected_features, indent=2),
                encoding="utf-8",
            )
            cv_metrics.to_csv(cv_metrics_path, index=False)
            cv_fold_metrics.to_csv(cv_fold_metrics_path, index=False)
            test_confusion_matrix.to_csv(test_confusion_matrix_path)
            test_confusion_matrix_plot.savefig(test_confusion_matrix_plot_path)
            with model_bundle_path.open("wb") as model_bundle_file:
                pickle.dump(model, model_bundle_file)

            if best_params is not None:
                best_params_path.write_text(
                    json.dumps(best_params, indent=2, default=str),
                    encoding="utf-8",
                )
            if optuna_trials is not None:
                optuna_trials.to_csv(optuna_trials_path, index=False)
            if optuna_fold_metrics is not None:
                optuna_fold_metrics.to_csv(optuna_fold_metrics_path, index=False)

            mlflow.pyfunc.log_model(
                name=artifact_model_name,
                python_model=_ModelBundlePyfuncModel(),
                artifacts={"model_bundle": str(model_bundle_path)},
            )
            mlflow.log_artifact(str(selected_features_path))
            mlflow.log_artifact(str(cv_metrics_path))
            mlflow.log_artifact(str(cv_fold_metrics_path))
            mlflow.log_artifact(str(test_confusion_matrix_path))
            mlflow.log_artifact(str(test_confusion_matrix_plot_path))
            if best_params is not None:
                mlflow.log_artifact(str(best_params_path))
            if optuna_trials is not None:
                mlflow.log_artifact(str(optuna_trials_path))
            if optuna_fold_metrics is not None:
                mlflow.log_artifact(str(optuna_fold_metrics_path))

        return pd.DataFrame(
            [
                {
                    "run_id": active_run.info.run_id,
                    "experiment_id": active_run.info.experiment_id,
                    "experiment_name": experiment.name,
                    "run_name": run_name,
                    "tracking_uri": mlflow_config["tracking_uri"],
                }
            ]
        )


def _log_metrics_frame(metrics_frame: pd.DataFrame) -> None:
    for metric_name, metric_value in metrics_frame.iloc[0].items():
        mlflow.log_metric(metric_name, float(metric_value))


def _log_test_metrics_frame(metrics_frame: pd.DataFrame) -> None:
    for metric_name, metric_value in metrics_frame.iloc[0].items():
        mlflow.log_metric(f"test_{metric_name}", float(metric_value))


def _log_optuna_trial_metadata(optuna_trials: pd.DataFrame) -> None:
    best_trials = optuna_trials.loc[optuna_trials["is_best"]]
    if best_trials.empty:
        raise ValueError("optuna_trials must contain one best trial.")
    best_trial = best_trials.iloc[0]
    mlflow.log_param("optimization_trial_count", len(optuna_trials))
    mlflow.log_param("best_trial_number", int(best_trial["trial_number"]))
    mlflow.log_metric("best_trial_cv_mean_f1", float(best_trial[PRIMARY_DEVELOPMENT_METRIC]))


def _mlflow_param_value(parameter_value: Any) -> str:
    if parameter_value is None:
        return "null"
    return str(parameter_value)


def _validate_training_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    _validate_feature_frame("X_train", X_train)
    _validate_label_series("y_train", y_train)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes must match.")


def _validate_feature_label_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    _validate_feature_frame("X_train", X_train)
    _validate_feature_frame("X_test", X_test)
    _validate_label_series("y_train", y_train)
    _validate_label_series("y_test", y_test)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes must match.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indexes must match.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test must have the same columns in the same order.")


def _validate_fold_artifacts(
    X_train: pd.DataFrame,
    X_evaluation: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    _validate_feature_frame("fold X_train", X_train)
    _validate_feature_frame("fold X_evaluation", X_evaluation)
    _validate_label_series("fold y_train", y_train)
    if len(X_train) != len(y_train):
        raise ValueError("Fold X_train and y_train must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Fold X_train and y_train indexes must match.")
    if list(X_train.columns) != list(X_evaluation.columns):
        raise ValueError("Fold train and evaluation features must have matching columns.")


def _validate_test_artifacts(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None,
) -> None:
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("model must expose predict and predict_proba methods.")
    _validate_feature_frame("X_test", X_test)
    _validate_label_series("y_test", y_test)
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indexes must match.")
    if selected_features is not None:
        _validate_selected_features(selected_features, X_test)


def _validate_feature_frame(artifact_name: str, feature_frame: pd.DataFrame) -> None:
    if not isinstance(feature_frame, pd.DataFrame):
        raise ValueError(f"{artifact_name} must be a pandas DataFrame.")
    if feature_frame.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    if feature_frame.shape[1] == 0:
        raise ValueError(f"{artifact_name} must contain at least one feature column.")
    non_numeric_columns = feature_frame.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"{artifact_name} contains non-numeric columns: {non_numeric_columns}")


def _validate_label_series(artifact_name: str, labels: pd.Series) -> None:
    if not isinstance(labels, pd.Series):
        raise ValueError(f"{artifact_name} must be a pandas Series.")
    if labels.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    unique_values = set(labels.dropna().unique())
    if not unique_values.issubset(set(CLASS_LABELS)):
        raise ValueError(f"{artifact_name} must contain only labels {CLASS_LABELS}.")


def _validate_selected_features(
    selected_features: list[str],
    feature_frame: pd.DataFrame,
) -> None:
    if not isinstance(selected_features, list):
        raise ValueError("selected_features must be a list.")
    if not selected_features:
        raise ValueError("selected_features must not be empty.")
    missing_columns = [feature for feature in selected_features if feature not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"selected_features are missing from X_test: {missing_columns}")


def _validate_mlflow_inputs(
    model: Any,
    selected_features: list[str],
    cv_metrics: pd.DataFrame,
    cv_fold_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    test_confusion_matrix_plot: Figure,
    best_params: dict[str, Any] | None,
    optuna_trials: pd.DataFrame | None,
    optuna_fold_metrics: pd.DataFrame | None,
) -> None:
    if not hasattr(model, "get_params"):
        raise ValueError("model must expose get_params for MLflow logging.")
    _validate_selected_features(selected_features, pd.DataFrame(columns=selected_features))
    _validate_cv_metrics_frame(cv_metrics)
    _validate_cv_fold_metrics_frame(cv_fold_metrics)
    _validate_test_metrics_frame(test_metrics)
    _validate_confusion_matrix_frame(test_confusion_matrix)
    if not isinstance(test_confusion_matrix_plot, Figure):
        raise ValueError("test_confusion_matrix_plot must be a matplotlib Figure.")
    if best_params is not None:
        _validate_random_forest_parameters(best_params)
    if optuna_trials is not None:
        _validate_optuna_trials_frame(optuna_trials)
    if optuna_fold_metrics is not None:
        _validate_optuna_fold_metrics_frame(optuna_fold_metrics)


def _validate_cv_metrics_frame(cv_metrics: pd.DataFrame) -> None:
    required_columns = {
        f"cv_{aggregate}_{metric_name}"
        for aggregate in ("mean", "std")
        for metric_name in METRIC_NAMES
    }
    _validate_metrics_dataframe("cv_metrics", cv_metrics, required_columns)
    if len(cv_metrics) != 1:
        raise ValueError("cv_metrics must contain exactly one summary row.")


def _validate_cv_fold_metrics_frame(cv_fold_metrics: pd.DataFrame) -> None:
    required_columns = {
        "fold",
        "train_row_count",
        "filtered_train_row_count",
        "validation_row_count",
        "selected_feature_count",
        *METRIC_NAMES,
    }
    _validate_metrics_dataframe("cv_fold_metrics", cv_fold_metrics, required_columns)


def _validate_optuna_trials_frame(optuna_trials: pd.DataFrame) -> None:
    required_columns = {
        "trial_number",
        "objective_value",
        "is_best",
        PRIMARY_DEVELOPMENT_METRIC,
    }
    _validate_metrics_dataframe("optuna_trials", optuna_trials, required_columns)
    if int(optuna_trials["is_best"].sum()) != 1:
        raise ValueError("optuna_trials must contain exactly one best trial.")


def _validate_optuna_fold_metrics_frame(optuna_fold_metrics: pd.DataFrame) -> None:
    required_columns = {"trial_number", *_cv_fold_output_columns()}
    _validate_metrics_dataframe(
        "optuna_fold_metrics",
        optuna_fold_metrics,
        required_columns,
    )


def _validate_test_metrics_frame(test_metrics: pd.DataFrame) -> None:
    _validate_metrics_dataframe("test_metrics", test_metrics, set(METRIC_NAMES))
    if len(test_metrics) != 1:
        raise ValueError("test_metrics must contain exactly one row.")


def _validate_metrics_dataframe(
    artifact_name: str,
    metrics_frame: pd.DataFrame,
    required_columns: set[str],
) -> None:
    if not isinstance(metrics_frame, pd.DataFrame):
        raise ValueError(f"{artifact_name} must be a pandas DataFrame.")
    if metrics_frame.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    missing_columns = required_columns.difference(metrics_frame.columns)
    if missing_columns:
        raise ValueError(f"{artifact_name} is missing columns: {sorted(missing_columns)}")


def _validate_confusion_matrix_frame(confusion_matrix_frame: pd.DataFrame) -> None:
    if not isinstance(confusion_matrix_frame, pd.DataFrame):
        raise ValueError("confusion matrix must be a pandas DataFrame.")
    if confusion_matrix_frame.shape != (2, 2):
        raise ValueError("confusion matrix must be a 2x2 DataFrame.")
    if _coerce_class_labels(confusion_matrix_frame.index) != CLASS_LABELS:
        raise ValueError(f"confusion matrix index must be {CLASS_LABELS}.")
    if _coerce_class_labels(confusion_matrix_frame.columns) != CLASS_LABELS:
        raise ValueError(f"confusion matrix columns must be {CLASS_LABELS}.")


def _coerce_class_labels(labels: pd.Index) -> list[int]:
    """Return binary labels as integers after CSV catalog round-trips."""
    try:
        return [int(label) for label in labels]
    except (TypeError, ValueError):
        return list(labels)
