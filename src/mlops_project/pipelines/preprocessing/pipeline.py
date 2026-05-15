"""Kedro pipeline definition for preprocessing."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the preprocessing pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.split_dataset,
                inputs=["water_potability", "params:preprocessing"],
                outputs=[
                    "X_train_split",
                    "X_validation_split",
                    "X_test_split",
                    "y_train_split",
                    "y_validation",
                    "y_test",
                ],
                name="split_dataset_node",
            ),
            node(
                func=nodes.remove_outliers,
                inputs=["X_train_split", "y_train_split", "params:preprocessing"],
                outputs=["X_train_filtered", "y_train"],
                name="remove_training_outliers_node",
            ),
            node(
                func=nodes.impute_features,
                inputs=["X_train_filtered", "X_validation_split", "X_test_split"],
                outputs=["X_train_imputed", "X_validation_imputed", "X_test_imputed"],
                name="impute_features_node",
            ),
            node(
                func=nodes.scale_features,
                inputs=["X_train_imputed", "X_validation_imputed", "X_test_imputed"],
                outputs=["X_train", "X_validation", "X_test"],
                name="scale_features_node",
            ),
        ]
    )
