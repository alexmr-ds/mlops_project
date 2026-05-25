"""Fold-local learned preprocessing for model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelPreprocessor:
    """Fit and apply learned preprocessing without leaking evaluation rows."""

    parameters: dict[str, Any]
    imputer: SimpleImputer | None = field(default=None, init=False)
    scaler: StandardScaler | None = field(default=None, init=False)
    selector: RFECV | None = field(default=None, init=False)
    selected_features: list[str] = field(default_factory=list, init=False)
    input_features: list[str] = field(default_factory=list, init=False)
    filtered_training_row_count: int = field(default=0, init=False)

    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit learned transforms and return transformed training rows."""
        _validate_training_artifacts(X_train, y_train)
        self.input_features = X_train.columns.tolist()

        filtered_X_train, filtered_y_train = remove_outliers_for_training(
            X_train,
            y_train,
            self.parameters,
        )
        self.filtered_training_row_count = len(filtered_X_train)

        self.imputer = SimpleImputer(strategy="mean")
        imputed_train = pd.DataFrame(
            self.imputer.fit_transform(filtered_X_train),
            columns=filtered_X_train.columns,
            index=filtered_X_train.index,
        )

        self.scaler = StandardScaler()
        scaled_train = pd.DataFrame(
            self.scaler.fit_transform(imputed_train),
            columns=imputed_train.columns,
            index=imputed_train.index,
        )

        selected_train = self._fit_select_features(scaled_train, filtered_y_train)
        return selected_train, filtered_y_train

    def transform(self, X_evaluation: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted learned transforms to evaluation rows."""
        self._validate_is_fitted()
        _validate_evaluation_features(X_evaluation, self.input_features)

        imputed_evaluation = pd.DataFrame(
            self.imputer.transform(X_evaluation),
            columns=X_evaluation.columns,
            index=X_evaluation.index,
        )
        scaled_evaluation = pd.DataFrame(
            self.scaler.transform(imputed_evaluation),
            columns=imputed_evaluation.columns,
            index=imputed_evaluation.index,
        )
        return scaled_evaluation.loc[:, self.selected_features].copy()

    def _fit_select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> pd.DataFrame:
        """Fit optional RFECV and return selected training features."""
        rfe_config = self.parameters["rfe"]
        if not rfe_config.get("enabled", True):
            self.selector = None
            self.selected_features = X_train.columns.tolist()
            return X_train.copy()

        fold_count = _inner_feature_selection_fold_count(
            y_train,
            requested_folds=int(rfe_config["cv_folds"]),
        )
        self.selector = RFECV(
            estimator=LogisticRegression(
                max_iter=rfe_config["logistic_max_iter"],
                random_state=rfe_config["random_state"],
            ),
            step=rfe_config["step"],
            cv=StratifiedKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=rfe_config["random_state"],
            ),
            scoring=rfe_config["scoring"],
            n_jobs=rfe_config["n_jobs"],
        )
        self.selector.fit(X_train, y_train)
        self.selected_features = X_train.columns[self.selector.support_].tolist()
        if not self.selected_features:
            raise ValueError("RFECV did not select any features.")
        return X_train.loc[:, self.selected_features].copy()

    def _validate_is_fitted(self) -> None:
        """Validate that fit_resample has run before transform."""
        if self.imputer is None or self.scaler is None or not self.selected_features:
            raise ValueError("ModelPreprocessor must be fitted before transform.")


def remove_outliers_for_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove outlier rows from training data only."""
    feature_columns = parameters.get("feature_columns")
    if feature_columns is None:
        candidate_columns = list(X_train.columns)
    else:
        missing_columns = [
            column for column in feature_columns if column not in X_train.columns
        ]
        if missing_columns:
            raise ValueError(
                "Configured outlier feature columns are missing from the training data: "
                f"{missing_columns}"
            )
        candidate_columns = list(feature_columns)

    threshold = float(parameters["zscore_threshold"])
    candidate_values = X_train.loc[:, candidate_columns]
    feature_means = candidate_values.mean(axis=0, skipna=True)
    feature_stds = candidate_values.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    z_score_frame = ((candidate_values - feature_means) / feature_stds).fillna(0.0)
    inlier_mask = (np.abs(z_score_frame) < threshold).all(axis=1)

    filtered_X_train = X_train.loc[inlier_mask].copy()
    filtered_y_train = y_train.loc[inlier_mask].copy()
    if filtered_X_train.empty:
        raise ValueError("Outlier removal removed every training row.")
    return filtered_X_train, filtered_y_train


def _inner_feature_selection_fold_count(y_train: pd.Series, requested_folds: int) -> int:
    """Return a valid inner CV fold count for RFECV."""
    if requested_folds < 2:
        raise ValueError("RFECV cv_folds must be at least 2.")
    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        raise ValueError("RFECV requires at least 2 rows in every class.")
    return min(requested_folds, min_class_count)


def _validate_training_artifacts(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Validate fit inputs for learned preprocessing."""
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be a pandas Series.")
    if X_train.empty:
        raise ValueError("X_train must not be empty.")
    if y_train.empty:
        raise ValueError("y_train must not be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes must match.")
    non_numeric_columns = X_train.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"X_train contains non-numeric columns: {non_numeric_columns}")


def _validate_evaluation_features(
    X_evaluation: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """Validate transform inputs for learned preprocessing."""
    if not isinstance(X_evaluation, pd.DataFrame):
        raise ValueError("X_evaluation must be a pandas DataFrame.")
    if X_evaluation.empty:
        raise ValueError("X_evaluation must not be empty.")
    if X_evaluation.columns.tolist() != expected_features:
        raise ValueError("X_evaluation columns must match fitted training columns.")
    non_numeric_columns = X_evaluation.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"X_evaluation contains non-numeric columns: {non_numeric_columns}"
        )
