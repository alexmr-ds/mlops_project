"""Kedro pipeline definition for preprocessing."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes, validation


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the preprocessing pipeline."""
    del kwargs
    return pipeline(
        [
            node(
                func=validation.validate_raw_data,
                inputs="water_potability",
                outputs="validated_water_potability",
                name="validate_raw_data_node",
            ),
            node(
                func=nodes.split_dataset,
                inputs=["validated_water_potability", "params:preprocessing"],
                outputs=[
                    "X_train_split",
                    "X_validation_split",
                    "X_test_split",
                    "y_train_split",
                    "y_validation_candidate",
                    "y_test_candidate",
                ],
                name="split_dataset_node",
            ),
            node(
                func=nodes.engineer_features,
                inputs=["X_train_split", "X_validation_split", "X_test_split"],
                outputs=[
                    "X_train_engineered",
                    "X_validation_engineered",
                    "X_test_engineered",
                ],
                name="engineer_features_node",
            ),
            node(
                func=nodes.remove_outliers,
                inputs=["X_train_engineered", "y_train_split", "params:preprocessing"],
                outputs=["X_train_filtered", "y_train_candidate"],
                name="remove_training_outliers_node",
            ),
            node(
                func=nodes.impute_features,
                inputs=["X_train_filtered", "X_validation_engineered", "X_test_engineered"],
                outputs=["X_train_imputed", "X_validation_imputed", "X_test_imputed"],
                name="impute_features_node",
            ),
            node(
                func=nodes.scale_features,
                inputs=["X_train_imputed", "X_validation_imputed", "X_test_imputed"],
                outputs=["X_train_scaled", "X_validation_scaled", "X_test_scaled"],
                name="scale_features_node",
            ),
            node(
                func=nodes.select_features_rfe,
                inputs=["X_train_scaled", "y_train_candidate", "params:preprocessing"],
                outputs=[
                    "X_train_candidate",
                    "selected_features_candidate",
                    "rfe_summary",
                ],
                name="select_features_rfe_node",
            ),
            node(
                func=nodes.apply_selected_features,
                inputs=[
                    "X_validation_scaled",
                    "X_test_scaled",
                    "selected_features_candidate",
                ],
                outputs=["X_validation_candidate", "X_test_candidate"],
                name="apply_selected_features_node",
            ),
            node(
                func=validation.validate_preprocessed_data,
                inputs=[
                    "X_train_candidate",
                    "X_validation_candidate",
                    "X_test_candidate",
                    "y_train_candidate",
                    "y_validation_candidate",
                    "y_test_candidate",
                    "selected_features_candidate",
                ],
                outputs=[
                    "X_train",
                    "X_validation",
                    "X_test",
                    "y_train",
                    "y_validation",
                    "y_test",
                    "selected_features",
                ],
                name="validate_preprocessed_data_node",
            ),
        ]
    )
