"""Pipeline registry for the Kedro project."""

from kedro.pipeline import Pipeline

from .pipelines import data_drift, modeling, preprocessing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    preprocessing_pipeline = preprocessing.create_pipeline()
    logistic_regression_pipeline = modeling.create_logistic_regression_pipeline()
    random_forest_pipeline = modeling.create_random_forest_pipeline()
    extra_trees_pipeline = modeling.create_extra_trees_pipeline()
    hist_gradient_boosting_pipeline = modeling.create_hist_gradient_boosting_pipeline()
    xgboost_pipeline = modeling.create_xgboost_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    data_drift_pipeline = data_drift.create_pipeline()
    return {
        "preprocessing": preprocessing_pipeline,
        "modeling_logistic_regression": logistic_regression_pipeline,
        "modeling_random_forest": random_forest_pipeline,
        "modeling_extra_trees": extra_trees_pipeline,
        "modeling_hist_gradient_boosting": hist_gradient_boosting_pipeline,
        "modeling_xgboost": xgboost_pipeline,
        "modeling": modeling_pipeline,
        "data_drift": data_drift_pipeline,
        "__default__": preprocessing_pipeline + modeling_pipeline,
    }
