"""Kedro pipeline definition for data drift detection."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the data drift detection pipeline.

    Two scenarios are checked:

    1. Training vs. test split -- a sanity baseline. X_train and X_test come
       from the same random split, so this should show little to no drift.
    2. Training vs. a simulated production sample -- X_test with a hypothetical
       shift in the raw measurements applied, recomputed through the same
       feature engineering used at training time. This is the scenario that
       resembles what drift monitoring is actually meant to catch, and we also
       score the trained Random Forest on it to see the metric degradation.
    """
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.detect_feature_drift,
                inputs=["X_train", "X_test", "params:data_drift"],
                outputs="drift_report",
                name="detect_feature_drift_node",
            ),
            node(
                func=nodes.simulate_production_drift,
                inputs=["X_test", "params:data_drift"],
                outputs="simulated_X_test",
                name="simulate_production_drift_node",
            ),
            node(
                func=nodes.detect_feature_drift,
                inputs=["X_train", "simulated_X_test", "params:data_drift"],
                outputs="simulated_drift_report",
                name="detect_simulated_drift_node",
            ),
            node(
                func=nodes.evaluate_model_under_simulated_drift,
                inputs=["random_forest_model", "simulated_X_test", "y_test"],
                outputs="simulated_drift_metrics",
                name="evaluate_model_under_simulated_drift_node",
            ),
        ]
    )
