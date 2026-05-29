"""Model training, cross-validation, and holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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


def cross_validate_and_train_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cross-validate, refit, and test one model family."""
    validate_feature_label_artifacts(X_train, X_test, y_train, y_test)
    cv_result = cross_validate_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )

    model, test_metrics, test_confusion_matrix, selected_features = train_evaluate_model(
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )

    return (
        model,
        cv_result.cv_metrics,
        cv_result.cv_fold_metrics,
        test_metrics,
        test_confusion_matrix,
        selected_features,
    )


def cross_validate_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> CrossValidationResult:
    """Evaluate one model family through training-set-only CV."""
    validate_training_artifacts(X_train, y_train)
    cross_validation_config = modeling_parameters.get("cross_validation", {})
    splitter = build_cross_validation_splitter(y_train, cross_validation_config)

    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        fold_X_train = X_train.iloc[train_positions]
        fold_y_train = y_train.iloc[train_positions]
        fold_X_validation = X_train.iloc[validation_positions]
        fold_y_validation = y_train.iloc[validation_positions]

        fitted_stack = fit_modeling_stack(
            model_name=model_name,
            X_train=fold_X_train,
            y_train=fold_y_train,
            X_evaluation=fold_X_validation,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=modeling_parameters,
        )
        fold_metrics = calculate_metrics(
            y_true=fold_y_validation,
            predictions=fitted_stack.model.predict(fitted_stack.evaluation_features),
            probabilities=predict_positive_class_probability(
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
    cv_metrics = summarize_cv_metrics(cv_fold_metrics)
    return CrossValidationResult(
        cv_metrics=cv_metrics,
        cv_fold_metrics=cv_fold_metrics,
    )


def train_evaluate_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit one model family and evaluate the final holdout split."""
    validate_feature_label_artifacts(X_train, X_test, y_train, y_test)
    final_stack = fit_modeling_stack(
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
        test_metrics,
        test_confusion_matrix,
        final_stack.selected_features,
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted model on the final holdout test split."""
    validate_test_artifacts(model, X_test, y_test, selected_features)

    evaluation_features = X_test
    if selected_features is not None and not isinstance(
        model, modeling_model_bundle.ModelBundle
    ):
        evaluation_features = X_test.loc[:, selected_features]

    predictions = model.predict(evaluation_features)
    probabilities = predict_positive_class_probability(model, evaluation_features)
    metrics_frame = pd.DataFrame([calculate_metrics(y_test, predictions, probabilities)])
    confusion_matrix_frame = build_confusion_matrix_frame(y_test, predictions)
    return metrics_frame, confusion_matrix_frame


def fit_modeling_stack(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_evaluation: pd.DataFrame,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> FittedModelingStack:
    """Fit fold-local learned preprocessing and estimator."""
    validate_fold_artifacts(X_train, X_evaluation, y_train)

    preprocessor = modeling_preprocessing.ModelPreprocessor(
        preprocessing_parameters,
    )
    selected_train, filtered_y_train = preprocessor.fit_resample(X_train, y_train)

    estimator = build_model(model_name, modeling_parameters)
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


def build_model(model_name: str, modeling_parameters: dict[str, Any]) -> Any:
    """Build an unfitted estimator for a supported model family."""
    if model_name == "logistic_regression":
        return LogisticRegression(**modeling_parameters["logistic_regression"])
    if model_name == "random_forest":
        return RandomForestClassifier(**modeling_parameters["random_forest"])
    raise ValueError(f"Unsupported model name: {model_name}")


def build_cross_validation_splitter(
    y_train: pd.Series,
    cross_validation_config: dict[str, Any],
) -> StratifiedKFold:
    """Build a valid stratified cross-validation splitter."""
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


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc": safe_roc_auc_score(y_true, probabilities),
    }


def safe_roc_auc_score(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Return NaN instead of failing when ROC AUC is undefined."""
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def predict_positive_class_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict the probability of the potable class."""
    probabilities = model.predict_proba(features)
    positive_class_index = list(model.classes_).index(1)
    return probabilities[:, positive_class_index]


def summarize_cv_metrics(cv_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold metrics into one CV metric row."""
    summary: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        metric_values = cv_fold_metrics[metric_name]
        summary[f"cv_mean_{metric_name}"] = float(metric_values.mean())
        summary[f"cv_std_{metric_name}"] = float(metric_values.std(ddof=0))
    return pd.DataFrame([summary])


def build_confusion_matrix_frame(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build a labeled binary confusion matrix frame."""
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
    return pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)


def cv_fold_output_columns() -> list[str]:
    """Return the canonical per-fold metric output columns."""
    return [
        "fold",
        "train_row_count",
        "filtered_train_row_count",
        "validation_row_count",
        "selected_feature_count",
        *METRIC_NAMES,
    ]


def validate_training_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Validate training feature and label artifacts."""
    validate_feature_frame("X_train", X_train)
    validate_label_series("y_train", y_train)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes must match.")


def validate_feature_label_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Validate train/test feature and label artifacts."""
    validate_feature_frame("X_train", X_train)
    validate_feature_frame("X_test", X_test)
    validate_label_series("y_train", y_train)
    validate_label_series("y_test", y_test)
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


def validate_fold_artifacts(
    X_train: pd.DataFrame,
    X_evaluation: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Validate one fold's train and evaluation artifacts."""
    validate_feature_frame("fold X_train", X_train)
    validate_feature_frame("fold X_evaluation", X_evaluation)
    validate_label_series("fold y_train", y_train)
    if len(X_train) != len(y_train):
        raise ValueError("Fold X_train and y_train must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Fold X_train and y_train indexes must match.")
    if list(X_train.columns) != list(X_evaluation.columns):
        raise ValueError("Fold train and evaluation features must have matching columns.")


def validate_test_artifacts(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None,
) -> None:
    """Validate artifacts used for final holdout evaluation."""
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("model must expose predict and predict_proba methods.")
    validate_feature_frame("X_test", X_test)
    validate_label_series("y_test", y_test)
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test indexes must match.")
    if selected_features is not None:
        validate_selected_features(selected_features, X_test)


def validate_feature_frame(artifact_name: str, feature_frame: pd.DataFrame) -> None:
    """Validate a numeric feature frame."""
    if not isinstance(feature_frame, pd.DataFrame):
        raise ValueError(f"{artifact_name} must be a pandas DataFrame.")
    if feature_frame.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    if feature_frame.shape[1] == 0:
        raise ValueError(f"{artifact_name} must contain at least one feature column.")
    non_numeric_columns = feature_frame.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"{artifact_name} contains non-numeric columns: {non_numeric_columns}")


def validate_label_series(artifact_name: str, labels: pd.Series) -> None:
    """Validate binary label values."""
    if not isinstance(labels, pd.Series):
        raise ValueError(f"{artifact_name} must be a pandas Series.")
    if labels.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    unique_values = set(labels.dropna().unique())
    if not unique_values.issubset(set(CLASS_LABELS)):
        raise ValueError(f"{artifact_name} must contain only labels {CLASS_LABELS}.")


def validate_selected_features(
    selected_features: list[str],
    feature_frame: pd.DataFrame,
) -> None:
    """Validate that selected features are non-empty and present."""
    if not isinstance(selected_features, list):
        raise ValueError("selected_features must be a list.")
    if not selected_features:
        raise ValueError("selected_features must not be empty.")
    missing_columns = [
        feature for feature in selected_features if feature not in feature_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"selected_features are missing from X_test: {missing_columns}")


def validate_confusion_matrix_frame(confusion_matrix_frame: pd.DataFrame) -> None:
    """Validate a binary confusion matrix frame."""
    if not isinstance(confusion_matrix_frame, pd.DataFrame):
        raise ValueError("confusion matrix must be a pandas DataFrame.")
    if confusion_matrix_frame.shape != (2, 2):
        raise ValueError("confusion matrix must be a 2x2 DataFrame.")
    if coerce_class_labels(confusion_matrix_frame.index) != CLASS_LABELS:
        raise ValueError(f"confusion matrix index must be {CLASS_LABELS}.")
    if coerce_class_labels(confusion_matrix_frame.columns) != CLASS_LABELS:
        raise ValueError(f"confusion matrix columns must be {CLASS_LABELS}.")


def validate_metrics_dataframe(
    artifact_name: str,
    metrics_frame: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Validate that a metric artifact has the expected columns."""
    if not isinstance(metrics_frame, pd.DataFrame):
        raise ValueError(f"{artifact_name} must be a pandas DataFrame.")
    if metrics_frame.empty:
        raise ValueError(f"{artifact_name} must not be empty.")
    missing_columns = required_columns.difference(metrics_frame.columns)
    if missing_columns:
        raise ValueError(f"{artifact_name} is missing columns: {sorted(missing_columns)}")


def coerce_class_labels(labels: pd.Index) -> list[int]:
    """Return binary labels as integers after CSV catalog round-trips."""
    try:
        return [int(label) for label in labels]
    except (TypeError, ValueError):
        return list(labels)
