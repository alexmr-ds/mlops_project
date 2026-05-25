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
                    "X_test_split",
                    "y_train_split",
                    "y_test_candidate",
                ],
                name="split_dataset_node",
            ),
            node(
                func=nodes.engineer_features,
                inputs=["X_train_split", "X_test_split"],
                outputs=[
                    "X_train_engineered",
                    "X_test_engineered",
                ],
                name="engineer_features_node",
            ),
            node(
                func=validation.validate_modeling_input_data,
                inputs=[
                    "X_train_engineered",
                    "X_test_engineered",
                    "y_train_split",
                    "y_test_candidate",
                ],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="validate_modeling_input_data_node",
            ),
        ]
    )
