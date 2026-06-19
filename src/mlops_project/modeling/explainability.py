"""SHAP-based feature importance for the best-performing model."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure


def compute_shap_values(
    model_bundle: Any,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for the positive class using TreeExplainer.

    The test features are passed through the model bundle's fitted preprocessor
    (imputation, scaling, feature selection) before SHAP values are computed,
    so the explanation reflects the exact inputs the estimator sees.
    """
    X_transformed = model_bundle.preprocessor.transform(X_test)
    feature_names = X_transformed.columns.tolist()

    explainer = shap.TreeExplainer(model_bundle.estimator)
    raw_shap = explainer.shap_values(X_transformed)

    # RandomForestClassifier returns one array per class; we explain class 1 (potable)
    if isinstance(raw_shap, list):
        shap_values = np.array(raw_shap[1])
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_values = raw_shap[:, :, 1]
    else:
        shap_values = np.array(raw_shap)

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_values, summary


def create_shap_summary_plot(
    shap_values: np.ndarray,
    shap_summary: pd.DataFrame,
) -> Figure:
    """Horizontal bar chart of mean |SHAP| per feature, sorted by importance."""
    # Reverse order so the most important feature appears at the top
    features = shap_summary["feature"].tolist()[::-1]
    values = shap_summary["mean_abs_shap"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.35)))
    bars = ax.barh(features, values, color="#4c72b0", edgecolor="white", linewidth=0.5)

    # Label each bar with its value for easy reading
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Mean |SHAP Value| (positive class)", fontsize=10)
    ax.set_title("Feature Importance — Random Forest (SHAP)", fontsize=12, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    return fig
