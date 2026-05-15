"""Node functions for Kedro preprocessing steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a labelled dataframe into train, optional validation, and test sets."""
    target_column = parameters["target_column"]
    test_size = float(parameters["test_size"])
    validation_size = float(parameters["validation_size"])
    random_state = int(parameters.get("random_state", 42))
    shuffle = bool(parameters.get("shuffle", True))
    use_stratify = bool(parameters.get("stratify", True))

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the input data.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size < 0 or validation_size >= 1:
        raise ValueError("validation_size must be in the range [0, 1).")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    features = data.drop(columns=target_column)
    target = data[target_column]
    stratify_values = target if use_stratify else None

    if validation_size == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_values,
        )
        return (
            X_train.copy(),
            features.iloc[0:0].copy(),
            X_test.copy(),
            y_train.copy(),
            target.iloc[0:0].copy(),
            y_test.copy(),
        )

    holdout_size = test_size + validation_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=holdout_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_values,
    )
    holdout_stratify = y_holdout if use_stratify else None
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_size / holdout_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=holdout_stratify,
    )

    return (
        X_train.copy(),
        X_validation.copy(),
        X_test.copy(),
        y_train.copy(),
        y_validation.copy(),
        y_test.copy(),
    )


def remove_outliers(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove training rows whose feature Z-scores exceed the configured threshold."""
    feature_columns = parameters.get("feature_columns") or X_train.columns.tolist()
    missing_columns = sorted(set(feature_columns) - set(X_train.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns for outlier detection: {missing_columns}")

    threshold = float(parameters.get("zscore_threshold", 3.0))
    z_scores = np.abs(zscore(X_train[feature_columns], nan_policy="omit"))
    if isinstance(z_scores, np.ndarray) and z_scores.ndim == 1:
        z_scores = z_scores.reshape(-1, 1)

    z_score_frame = pd.DataFrame(z_scores, index=X_train.index, columns=feature_columns)
    outlier_mask = z_score_frame.gt(threshold).any(axis=1)
    keep_mask = ~outlier_mask.fillna(False)

    return X_train.loc[keep_mask].copy(), y_train.loc[keep_mask].copy()


def scale_features(
    X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a standard scaler on train data and transform every feature split."""
    scaler = StandardScaler()
    scaled_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    scaled_validation = _transform_split(scaler, X_validation)
    scaled_test = _transform_split(scaler, X_test)
    return scaled_train, scaled_validation, scaled_test


def impute_features(
    X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a mean imputer on train data and transform every feature split."""
    imputer = SimpleImputer(strategy="mean")
    imputed_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    imputed_validation = _transform_split(imputer, X_validation)
    imputed_test = _transform_split(imputer, X_test)
    return imputed_train, imputed_validation, imputed_test


def _transform_split(scaler: StandardScaler, split: pd.DataFrame) -> pd.DataFrame:
    """Transform a dataframe while preserving an empty optional validation split."""
    if split.empty:
        return split.copy()

    return pd.DataFrame(
        scaler.transform(split),
        columns=split.columns,
        index=split.index,
    )
