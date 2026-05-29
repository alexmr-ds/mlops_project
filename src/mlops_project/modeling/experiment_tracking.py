"""MLflow experiment tracking for fitted model bundles."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mlops_project.modeling import evaluation as modeling_evaluation
from mlops_project.modeling import optimization as modeling_optimization

LOGISTIC_REGRESSION_MODEL_ARTIFACT_NAME = "logistic_regression_model"
RANDOM_FOREST_MODEL_ARTIFACT_NAME = "random_forest_model"
RANDOM_FOREST_RUN_NAME = "random_forest_nonlinear_probe"


class ModelBundlePyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc adapter for persisted model bundles."""

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        del context
        return self._model_bundle.predict(model_input)

    def load_context(self, context: Any) -> None:
        """Load the bundled model artifact before serving predictions."""
        model_bundle_path = Path(context.artifacts["model_bundle"])
        with model_bundle_path.open("rb") as model_bundle_file:
            self._model_bundle = pickle.load(model_bundle_file)


def log_cv_model_to_mlflow(
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
    optuna_search_space: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Log CV, final holdout, and model bundle artifacts to MLflow."""
    validate_mlflow_inputs(
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
        optuna_search_space,
    )

    mlflow_config = parameters["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    experiment = mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(build_run_tags(parameters, model_name, optuna_trials))
        mlflow.log_params(
            {
                "model_name": model_name,
                "primary_development_metric": (
                    modeling_evaluation.PRIMARY_DEVELOPMENT_METRIC
                ),
                "selected_feature_count": len(selected_features),
            }
        )
        mlflow.log_params(model.get_params())
        if best_params is not None:
            mlflow.log_params(
                {
                    f"best_{parameter_name}": mlflow_param_value(parameter_value)
                    for parameter_name, parameter_value in best_params.items()
                }
            )
        if optuna_trials is not None:
            log_optuna_trial_metadata(optuna_trials)
        log_metrics_frame(cv_metrics)
        log_test_metrics_frame(test_metrics)

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
            optuna_search_space_path = temporary_path / "optuna_search_space.json"

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
            if optuna_search_space is not None:
                optuna_search_space_path.write_text(
                    json.dumps(optuna_search_space, indent=2, default=str),
                    encoding="utf-8",
                )

            mlflow.pyfunc.log_model(
                name=artifact_model_name,
                python_model=ModelBundlePyfuncModel(),
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
            if optuna_search_space is not None:
                mlflow.log_artifact(str(optuna_search_space_path))

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


def log_metrics_frame(metrics_frame: pd.DataFrame) -> None:
    """Log one metrics frame row to MLflow."""
    for metric_name, metric_value in metrics_frame.iloc[0].items():
        mlflow.log_metric(metric_name, float(metric_value))


def log_test_metrics_frame(metrics_frame: pd.DataFrame) -> None:
    """Log final holdout metrics with a test prefix."""
    for metric_name, metric_value in metrics_frame.iloc[0].items():
        mlflow.log_metric(f"test_{metric_name}", float(metric_value))


def build_run_tags(
    parameters: dict[str, Any],
    model_name: str,
    optuna_trials: pd.DataFrame | None,
) -> dict[str, str]:
    """Build stable MLflow UI tags for model development strategy."""
    cross_validation_config = parameters.get("cross_validation", {})
    n_splits = int(cross_validation_config.get("n_splits", 5))
    tags = {"cv_strategy": f"stratified_kfold_{n_splits}"}

    optimization_config = parameters.get("random_forest_optimization", {})
    if (
        model_name == "random_forest"
        and optuna_trials is not None
        and bool(optimization_config.get("enabled", True))
    ):
        sampler = str(optimization_config.get("sampler", "tpe"))
        tags["tuning_strategy"] = f"optuna_{sampler}"

    return tags


def log_optuna_trial_metadata(optuna_trials: pd.DataFrame) -> None:
    """Log compact Optuna trial metadata to the active MLflow run."""
    best_trials = optuna_trials.loc[optuna_trials["is_best"]]
    if best_trials.empty:
        raise ValueError("optuna_trials must contain one best trial.")
    best_trial = best_trials.iloc[0]
    mlflow.log_param("optimization_trial_count", len(optuna_trials))
    mlflow.log_param("best_trial_number", int(best_trial["trial_number"]))
    mlflow.log_metric(
        "best_trial_cv_mean_f1",
        float(best_trial[modeling_evaluation.PRIMARY_DEVELOPMENT_METRIC]),
    )


def mlflow_param_value(parameter_value: Any) -> str:
    """Convert a parameter to an MLflow-compatible string."""
    if parameter_value is None:
        return "null"
    return str(parameter_value)


def validate_mlflow_inputs(
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
    optuna_search_space: dict[str, Any] | None,
) -> None:
    """Validate artifacts before MLflow logging."""
    if not hasattr(model, "get_params"):
        raise ValueError("model must expose get_params for MLflow logging.")
    modeling_evaluation.validate_selected_features(
        selected_features,
        pd.DataFrame(columns=selected_features),
    )
    validate_cv_metrics_frame(cv_metrics)
    validate_cv_fold_metrics_frame(cv_fold_metrics)
    validate_test_metrics_frame(test_metrics)
    modeling_evaluation.validate_confusion_matrix_frame(test_confusion_matrix)
    if not isinstance(test_confusion_matrix_plot, Figure):
        raise ValueError("test_confusion_matrix_plot must be a matplotlib Figure.")
    if best_params is not None:
        modeling_optimization.validate_random_forest_parameters(best_params)
    if optuna_trials is not None:
        validate_optuna_trials_frame(optuna_trials)
    if optuna_fold_metrics is not None:
        validate_optuna_fold_metrics_frame(optuna_fold_metrics)
    if optuna_search_space is not None:
        validate_optuna_search_space(optuna_search_space)


def validate_cv_metrics_frame(cv_metrics: pd.DataFrame) -> None:
    """Validate a CV summary metrics artifact."""
    required_columns = {
        f"cv_{aggregate}_{metric_name}"
        for aggregate in ("mean", "std")
        for metric_name in modeling_evaluation.METRIC_NAMES
    }
    modeling_evaluation.validate_metrics_dataframe("cv_metrics", cv_metrics, required_columns)
    if len(cv_metrics) != 1:
        raise ValueError("cv_metrics must contain exactly one summary row.")


def validate_cv_fold_metrics_frame(cv_fold_metrics: pd.DataFrame) -> None:
    """Validate a CV fold metrics artifact."""
    required_columns = set(modeling_evaluation.cv_fold_output_columns())
    modeling_evaluation.validate_metrics_dataframe(
        "cv_fold_metrics",
        cv_fold_metrics,
        required_columns,
    )


def validate_optuna_trials_frame(optuna_trials: pd.DataFrame) -> None:
    """Validate an Optuna trial summary artifact."""
    required_columns = {
        "trial_number",
        "objective_value",
        "is_best",
        modeling_evaluation.PRIMARY_DEVELOPMENT_METRIC,
    }
    modeling_evaluation.validate_metrics_dataframe(
        "optuna_trials",
        optuna_trials,
        required_columns,
    )
    if int(optuna_trials["is_best"].sum()) != 1:
        raise ValueError("optuna_trials must contain exactly one best trial.")


def validate_optuna_fold_metrics_frame(optuna_fold_metrics: pd.DataFrame) -> None:
    """Validate an Optuna fold metrics artifact."""
    required_columns = {"trial_number", *modeling_evaluation.cv_fold_output_columns()}
    modeling_evaluation.validate_metrics_dataframe(
        "optuna_fold_metrics",
        optuna_fold_metrics,
        required_columns,
    )


def validate_test_metrics_frame(test_metrics: pd.DataFrame) -> None:
    """Validate a final holdout metrics artifact."""
    modeling_evaluation.validate_metrics_dataframe(
        "test_metrics",
        test_metrics,
        set(modeling_evaluation.METRIC_NAMES),
    )
    if len(test_metrics) != 1:
        raise ValueError("test_metrics must contain exactly one row.")


def validate_optuna_search_space(optuna_search_space: dict[str, Any]) -> None:
    """Validate an Optuna search space artifact payload."""
    if not isinstance(optuna_search_space, dict) or not optuna_search_space:
        raise ValueError("optuna_search_space must be a non-empty mapping.")
