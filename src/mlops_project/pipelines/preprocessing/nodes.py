"""Node functions for Kedro preprocessing steps."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

EPSILON = 1e-6


def split_dataset(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a labelled dataframe into train and final holdout test sets."""
    target_column = parameters["target_column"]
    test_size = float(parameters["test_size"])
    random_state = int(parameters.get("random_state", 73))
    shuffle = bool(parameters.get("shuffle", True))
    use_stratify = bool(parameters.get("stratify", True))

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the input data.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    features = data.drop(columns=target_column)
    target = data[target_column]
    stratify_values = target if use_stratify else None

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
        X_test.copy(),
        y_train.copy(),
        y_test.copy(),
    )


def engineer_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add deterministic engineered features to train and test splits."""
    return (
        _engineer_feature_frame(X_train),
        _engineer_feature_frame(X_test),
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
