"""Node functions for Kedro preprocessing steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-6


def split_dataset(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a labelled dataframe into train, optional validation, and test sets."""
    target_column = parameters["target_column"]
    test_size = float(parameters["test_size"])
    validation_size = float(parameters["validation_size"])
    random_state = int(parameters.get("random_state", 73))
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


def engineer_features(
    X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add deterministic engineered features to every feature split."""
    return (
        _engineer_feature_frame(X_train),
        _engineer_feature_frame(X_validation),
        _engineer_feature_frame(X_test),
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


def select_features_rfe(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Fit RFECV on scaled training data and return the selected training matrix."""
    rfe_parameters = parameters.get("rfe", {})
    enabled = bool(rfe_parameters.get("enabled", True))
    random_state = int(rfe_parameters.get("random_state", 73))
    logistic_max_iter = int(rfe_parameters.get("logistic_max_iter", 5000))
    cv_folds = int(rfe_parameters.get("cv_folds", 5))
    step = int(rfe_parameters.get("step", 1))
    scoring = str(rfe_parameters.get("scoring", "roc_auc"))
    n_jobs = int(rfe_parameters.get("n_jobs", -1))

    if not enabled:
        selected_features = X_train.columns.tolist()
        return X_train.copy(), selected_features

    estimator = LogisticRegression(max_iter=logistic_max_iter, random_state=random_state)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
    )
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_].tolist()
    selected_train = X_train.loc[:, selected_features].copy()
    return selected_train, selected_features


def apply_selected_features(
    X_validation: pd.DataFrame, X_test: pd.DataFrame, selected_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project holdout splits onto the training-selected feature set."""
    return (
        _select_split_columns(X_validation, selected_features),
        _select_split_columns(X_test, selected_features),
    )


def _engineer_feature_frame(split: pd.DataFrame) -> pd.DataFrame:
    """Create the configured derived features for one split."""
    if split.empty:
        return split.copy()

    engineered = split.copy()
    engineered["conductivity_solids_ratio"] = engineered["Conductivity"] / (
        engineered["Solids"] + EPSILON
    )
    engineered["turbidity_solids_ratio"] = engineered["Turbidity"] / (
        engineered["Solids"] + EPSILON
    )
    engineered["hardness_conductivity_ratio"] = engineered["Hardness"] / (
        engineered["Conductivity"] + EPSILON
    )
    engineered["hardness_solids_ratio"] = engineered["Hardness"] / (
        engineered["Solids"] + EPSILON
    )
    engineered["sulfate_hardness_ratio"] = engineered["Sulfate"] / (
        engineered["Hardness"] + EPSILON
    )
    engineered["tds_conductivity_ratio"] = engineered["Solids"] / (
        engineered["Conductivity"] + EPSILON
    )
    engineered["chloramines_ph_interaction"] = engineered["Chloramines"] * engineered["ph"]
    engineered["ph_hardness_interaction"] = engineered["ph"] * engineered["Hardness"]
    engineered["organic_trihalo_interaction"] = (
        engineered["Organic_carbon"] * engineered["Trihalomethanes"]
    )
    engineered["turbidity_organic_interaction"] = (
        engineered["Turbidity"] * engineered["Organic_carbon"]
    )
    engineered["trihalo_formation_risk"] = engineered["Organic_carbon"] / (
        engineered["Trihalomethanes"] + EPSILON
    )
    engineered["disinfection_stress"] = engineered["Sulfate"] + engineered["Chloramines"]
    engineered["dbp_precursor_load"] = (
        engineered["Organic_carbon"] + (engineered["Trihalomethanes"] / 10)
    )
    engineered["total_oxidant_stress"] = engineered["Chloramines"] + (
        engineered["Sulfate"] / 10
    )
    engineered["solids_sulfate_diff"] = engineered["Solids"] - engineered["Sulfate"]
    engineered["turbidity_trihalo_risk"] = (
        (engineered["Turbidity"] > 5) & (engineered["Trihalomethanes"] > 80)
    ).astype(int)
    engineered["risk_score"] = (
        ((engineered["ph"] < 6.5) | (engineered["ph"] > 8.5)).astype(int)
        + (engineered["Turbidity"] > 5).astype(int)
        + (engineered["Trihalomethanes"] > 80).astype(int)
        + (engineered["Chloramines"] > 10).astype(int)
        + (engineered["Sulfate"] > 400).astype(int)
    )
    engineered["expanded_risk_score"] = (
        engineered["risk_score"]
        + (engineered["Sulfate"] > 250).astype(int)
        + (engineered["Hardness"] > 200).astype(int)
        + (engineered["Conductivity"] > 400).astype(int)
    ).astype(int)
    engineered["ph_safe_range"] = (
        (engineered["ph"] >= 6.5) & (engineered["ph"] <= 8.5)
    ).astype(int)
    engineered["high_turbidity"] = (engineered["Turbidity"] > 5).astype(int)
    engineered["high_sulfate"] = (engineered["Sulfate"] > 250).astype(int)
    engineered["high_chloramines"] = (engineered["Chloramines"] > 4).astype(int)
    engineered["high_hardness"] = (engineered["Hardness"] > 200).astype(int)
    return engineered


def _transform_split(transformer: Any, split: pd.DataFrame) -> pd.DataFrame:
    """Transform a dataframe while preserving an empty optional validation split."""
    if split.empty:
        return split.copy()

    return pd.DataFrame(
        transformer.transform(split),
        columns=split.columns,
        index=split.index,
    )


def _select_split_columns(split: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Return one split restricted to the ordered selected feature list."""
    if split.empty:
        return pd.DataFrame(columns=selected_features, index=split.index)

    return split.loc[:, selected_features].copy()
