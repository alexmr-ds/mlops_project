"""Node functions for LogisticRegression baseline modeling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

CLASS_LABELS = [0, 1]
PRIMARY_DEVELOPMENT_METRIC = "validation_f1"


def train_logistic_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    parameters: dict[str, Any],
) -> LogisticRegression:
    """Train the first LogisticRegression baseline."""
    _validate_feature_label_artifacts(
        split_name="train",
        features=X_train,
        labels=y_train,
        selected_features=selected_features,
    )

    model_parameters = parameters.get("logistic_regression", {})
    model = LogisticRegression(
        max_iter=int(model_parameters.get("max_iter", 1000)),
        solver=str(model_parameters.get("solver", "lbfgs")),
        random_state=int(model_parameters.get("random_state", 73)),
    )
    model.fit(X_train.loc[:, selected_features], y_train)
    return model


def evaluate_validation_model(
    model: LogisticRegression,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline on the development validation split."""
    return evaluate_model(
        model=model,
        features=X_validation,
        labels=y_validation,
        selected_features=selected_features,
        split_name="validation",
    )


def evaluate_test_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline on the final holdout test split."""
    return evaluate_model(
        model=model,
        features=X_test,
        labels=y_test,
        selected_features=selected_features,
        split_name="test",
    )


def evaluate_model(
    model: LogisticRegression,
    features: pd.DataFrame,
    labels: pd.Series,
    selected_features: list[str],
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metrics and confusion matrix for one model evaluation split."""
    _validate_model(model)
    _validate_feature_label_artifacts(
        split_name=split_name,
        features=features,
        labels=labels,
        selected_features=selected_features,
    )

    split_features = features.loc[:, selected_features]
    predicted_labels = model.predict(split_features)
    predicted_probabilities = model.predict_proba(split_features)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(labels, predicted_labels),
                "precision": precision_score(labels, predicted_labels, zero_division=0),
                "recall": recall_score(labels, predicted_labels, zero_division=0),
                "f1": f1_score(labels, predicted_labels, zero_division=0),
                "f1_weighted": f1_score(
                    labels,
                    predicted_labels,
                    average="weighted",
                    zero_division=0,
                ),
                "roc_auc": _score_roc_auc(labels, predicted_probabilities, split_name),
            }
        ]
    )
    matrix = pd.DataFrame(
        confusion_matrix(labels, predicted_labels, labels=CLASS_LABELS),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    matrix.index.name = "actual"
    return metrics, matrix


def create_validation_confusion_matrix_plot(confusion_matrix_frame: pd.DataFrame) -> Figure:
    """Create the validation confusion matrix figure."""
    return create_confusion_matrix_plot(confusion_matrix_frame, split_name="validation")


def create_test_confusion_matrix_plot(confusion_matrix_frame: pd.DataFrame) -> Figure:
    """Create the test confusion matrix figure."""
    return create_confusion_matrix_plot(confusion_matrix_frame, split_name="test")


def create_confusion_matrix_plot(
    confusion_matrix_frame: pd.DataFrame, split_name: str
) -> Figure:
    """Create a compact matplotlib heatmap for a confusion matrix."""
    _validate_confusion_matrix(confusion_matrix_frame, split_name)

    matrix_values = confusion_matrix_frame[["predicted_0", "predicted_1"]].to_numpy()
    figure, axis = plt.subplots(figsize=(4, 3))
    image = axis.imshow(matrix_values, cmap="Blues")
    axis.set_title(f"{split_name.title()} confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks([0, 1], labels=["0", "1"])
    axis.set_yticks([0, 1], labels=["0", "1"])

    for row_index in range(matrix_values.shape[0]):
        for column_index in range(matrix_values.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix_values[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def log_model_to_mlflow(
    model: LogisticRegression,
    selected_features: list[str],
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    validation_confusion_matrix: pd.DataFrame,
    test_confusion_matrix: pd.DataFrame,
    validation_confusion_matrix_plot: Figure,
    test_confusion_matrix_plot: Figure,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Log the baseline model and evaluation artifacts to one MLflow run."""
    _validate_model(model)
    _validate_selected_features(selected_features)
    _validate_metrics_frame(validation_metrics, "validation")
    _validate_metrics_frame(test_metrics, "test")
    _validate_confusion_matrix(validation_confusion_matrix, "validation")
    _validate_confusion_matrix(test_confusion_matrix, "test")

    mlflow_parameters = parameters.get("mlflow", {})
    tracking_uri = str(mlflow_parameters.get("tracking_uri", "mlruns"))
    experiment_name = str(
        mlflow_parameters.get("experiment_name", "water_potability_modeling")
    )
    run_name = str(mlflow_parameters.get("run_name", "logistic_regression_baseline"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("primary_development_metric", PRIMARY_DEVELOPMENT_METRIC)
        mlflow.log_param("selected_feature_count", len(selected_features))
        mlflow.log_params(model.get_params())
        _log_split_metrics("validation", validation_metrics)
        _log_split_metrics("test", test_metrics)
        mlflow.sklearn.log_model(model, name="logistic_regression_model")

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = Path(temporary_directory)
            _write_selected_features_artifact(selected_features, artifact_dir)
            _write_confusion_matrix_artifact(
                validation_confusion_matrix,
                artifact_dir / "validation_confusion_matrix.csv",
            )
            _write_confusion_matrix_artifact(
                test_confusion_matrix,
                artifact_dir / "test_confusion_matrix.csv",
            )
            validation_confusion_matrix_plot.savefig(
                artifact_dir / "validation_confusion_matrix.png",
                bbox_inches="tight",
                dpi=150,
            )
            test_confusion_matrix_plot.savefig(
                artifact_dir / "test_confusion_matrix.png",
                bbox_inches="tight",
                dpi=150,
            )
            mlflow.log_artifacts(str(artifact_dir))

        run_info = run.info

    return pd.DataFrame(
        [
            {
                "run_id": run_info.run_id,
                "experiment_id": run_info.experiment_id,
                "experiment_name": experiment_name,
                "run_name": run_name,
                "tracking_uri": tracking_uri,
            }
        ]
    )


def _validate_model(model: LogisticRegression) -> None:
    """Raise if the model artifact is missing or the wrong type."""
    if not isinstance(model, LogisticRegression):
        raise ValueError("model must be a fitted LogisticRegression instance.")
    if not hasattr(model, "classes_"):
        raise ValueError("model must be a fitted LogisticRegression instance.")


def _validate_feature_label_artifacts(
    split_name: str,
    features: pd.DataFrame,
    labels: pd.Series,
    selected_features: list[str],
) -> None:
    """Validate model input artifacts before training or evaluation."""
    if not isinstance(features, pd.DataFrame):
        raise ValueError(f"{split_name} features must be a pandas DataFrame.")
    if not isinstance(labels, pd.Series):
        raise ValueError(f"{split_name} labels must be a pandas Series.")
    _validate_selected_features(selected_features)
    if features.empty:
        raise ValueError(f"{split_name} features must not be empty.")
    if labels.empty:
        raise ValueError(f"{split_name} labels must not be empty.")
    if len(features) != len(labels):
        raise ValueError(
            f"{split_name} X/y row counts must match "
            f"(features={len(features)}, labels={len(labels)})."
        )
    if not features.index.equals(labels.index):
        raise ValueError(f"{split_name} X/y indexes must match.")

    missing_features = sorted(set(selected_features) - set(features.columns))
    if missing_features:
        raise ValueError(
            f"{split_name} features are missing selected_features: {missing_features}"
        )


def _validate_selected_features(selected_features: list[str]) -> None:
    """Raise if the selected feature schema is missing or malformed."""
    if not isinstance(selected_features, list) or not selected_features:
        raise ValueError("selected_features must be a non-empty list.")
    if not all(isinstance(feature, str) for feature in selected_features):
        raise ValueError("selected_features must contain only strings.")
    if len(set(selected_features)) != len(selected_features):
        raise ValueError("selected_features must not contain duplicates.")


def _score_roc_auc(
    labels: pd.Series, predicted_probabilities: np.ndarray, split_name: str
) -> float:
    """Return ROC AUC or raise a clear error for one-class splits."""
    if labels.nunique() < 2:
        raise ValueError(f"{split_name} labels must contain both classes for roc_auc.")
    return roc_auc_score(labels, predicted_probabilities)


def _validate_metrics_frame(metrics: pd.DataFrame, split_name: str) -> None:
    """Raise if a persisted metrics frame is missing required columns."""
    required_columns = {"accuracy", "precision", "recall", "f1", "f1_weighted", "roc_auc"}
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError(f"{split_name} metrics must be a non-empty pandas DataFrame.")
    missing_columns = sorted(required_columns - set(metrics.columns))
    if missing_columns:
        raise ValueError(f"{split_name} metrics are missing columns: {missing_columns}")


def _validate_confusion_matrix(
    confusion_matrix_frame: pd.DataFrame, split_name: str
) -> None:
    """Raise if a confusion matrix frame is malformed."""
    expected_columns = ["predicted_0", "predicted_1"]
    if not isinstance(confusion_matrix_frame, pd.DataFrame):
        raise ValueError(f"{split_name} confusion matrix must be a pandas DataFrame.")
    if confusion_matrix_frame.shape != (2, 2):
        raise ValueError(f"{split_name} confusion matrix must have shape 2x2.")
    if confusion_matrix_frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"{split_name} confusion matrix columns must be {expected_columns}."
        )


def _metric_value(metrics: pd.DataFrame, metric_name: str) -> float:
    """Return one scalar metric from a one-row metrics dataframe."""
    return float(metrics.iloc[0][metric_name])


def _log_split_metrics(split_name: str, metrics: pd.DataFrame) -> None:
    """Log one split's metrics using prefixed MLflow metric names."""
    for metric_name in ["accuracy", "precision", "recall", "f1", "f1_weighted", "roc_auc"]:
        mlflow.log_metric(f"{split_name}_{metric_name}", _metric_value(metrics, metric_name))


def _write_selected_features_artifact(
    selected_features: list[str], artifact_dir: Path
) -> None:
    """Write selected features as a simple JSON artifact."""
    selected_features_path = artifact_dir / "selected_features.json"
    selected_features_path.write_text(
        json.dumps(selected_features, indent=2),
        encoding="utf-8",
    )


def _write_confusion_matrix_artifact(
    confusion_matrix_frame: pd.DataFrame, artifact_path: Path
) -> None:
    """Write a confusion matrix artifact with the same persisted CSV shape."""
    confusion_matrix_frame.to_csv(artifact_path, index=True)
