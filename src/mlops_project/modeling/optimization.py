"""Hyperparameter optimization evaluation for model training."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from mlops_project.modeling import evaluation as modeling_evaluation

PRIMARY_DEVELOPMENT_METRIC = modeling_evaluation.PRIMARY_DEVELOPMENT_METRIC
OPTUNA_TRIAL_PARAM_PREFIX = "param_"
TUNED_MODEL_NAMES = (
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
)
BOOTSTRAP_MAX_SAMPLE_MODELS = {"random_forest", "extra_trees"}
N_ESTIMATOR_MODELS = {"random_forest", "extra_trees", "xgboost"}


def tune_random_forest_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune random forest hyperparameters with training-set-only CV."""
    return tune_model_hyperparameters(
        model_name="random_forest",
        X_train=X_train,
        y_train=y_train,
        preprocessing_parameters=preprocessing_parameters,
        modeling_parameters=modeling_parameters,
    )


def tune_model_hyperparameters(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_parameters: dict[str, Any],
    modeling_parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune one supported model family with training-set-only CV."""
    validate_tuned_model_name(model_name)
    modeling_evaluation.validate_training_artifacts(X_train, y_train)
    optimization_config = model_optimization_config(model_name, modeling_parameters)
    trial_fold_frames: list[pd.DataFrame] = []

    if not bool(optimization_config.get("enabled", True)):
        best_params = resolve_model_parameters(modeling_parameters, model_name)
        cv_result = modeling_evaluation.cross_validate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=with_model_parameters(
                modeling_parameters,
                model_name,
                best_params,
            ),
        )
        trial_frame, trial_fold_frame = build_disabled_optimization_artifacts(
            best_params,
            cv_result,
        )
        return (
            best_params,
            cv_result.cv_metrics,
            cv_result.cv_fold_metrics,
            trial_frame,
            trial_fold_frame,
        )

    sampler = build_optuna_sampler(optimization_config)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        trial_params = sample_model_parameters(trial, optimization_config)
        model_params = resolve_model_parameters(
            modeling_parameters,
            model_name,
            selected_params=trial_params,
        )
        cv_result = modeling_evaluation.cross_validate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            preprocessing_parameters=preprocessing_parameters,
            modeling_parameters=with_model_parameters(
                modeling_parameters,
                model_name,
                model_params,
            ),
        )
        objective_value = float(cv_result.cv_metrics.loc[0, PRIMARY_DEVELOPMENT_METRIC])
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("cv_metrics", cv_result.cv_metrics.iloc[0].to_dict())
        trial_fold_frames.append(
            add_trial_context_to_fold_metrics(
                cv_result.cv_fold_metrics,
                trial.number,
                model_params,
            )
        )
        return objective_value

    study.optimize(
        objective,
        n_trials=int(optimization_config["n_trials"]),
        catch=(),
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["model_params"])
    best_cv_metrics = pd.DataFrame([best_trial.user_attrs["cv_metrics"]])
    trial_frame = build_optuna_trial_summary_frame(study)
    trial_fold_frame = pd.concat(trial_fold_frames, ignore_index=True)
    best_fold_metrics = trial_fold_frame.loc[
        trial_fold_frame["trial_number"] == best_trial.number,
        modeling_evaluation.cv_fold_output_columns(),
    ].reset_index(drop=True)
    return (
        best_params,
        best_cv_metrics,
        best_fold_metrics,
        trial_frame,
        trial_fold_frame,
    )


def random_forest_optimization_config(
    modeling_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return validated RandomForest optimization parameters."""
    return model_optimization_config("random_forest", modeling_parameters)


def model_optimization_config(
    model_name: str,
    modeling_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return validated optimization parameters for one model family."""
    validate_tuned_model_name(model_name)
    config_name = optimization_config_name(model_name)
    config = modeling_parameters.get(config_name, {})
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a mapping.")
    if "n_trials" not in config:
        raise ValueError(f"{config_name}.n_trials is required.")
    if int(config["n_trials"]) < 1:
        raise ValueError(f"{config_name}.n_trials must be at least 1.")
    if str(config.get("objective_metric", "f1")) != "f1":
        raise ValueError(f"{config_name} currently supports only f1.")
    if bool(config.get("enabled", True)):
        search_space = config.get("search_space")
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError(f"{config_name}.search_space must not be empty.")
    return config


def optimization_config_name(model_name: str) -> str:
    """Return the modeling parameter key for one model family's optimization config."""
    return f"{model_name}_optimization"


def validate_tuned_model_name(model_name: str) -> None:
    """Validate that a model family supports Optuna tuning."""
    if model_name not in TUNED_MODEL_NAMES:
        raise ValueError(f"Unsupported tuned model name: {model_name}")


def build_optuna_sampler(
    optimization_config: dict[str, Any],
) -> optuna.samplers.BaseSampler:
    """Build the configured Optuna sampler."""
    sampler_name = str(optimization_config.get("sampler", "tpe"))
    random_state = int(optimization_config.get("random_state", 73))
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=random_state)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


def sample_random_forest_parameters(
    trial: optuna.Trial,
    optimization_config: dict[str, Any],
) -> dict[str, Any]:
    """Sample one RandomForest parameter set from the configured search space."""
    return sample_model_parameters(trial, optimization_config)


def sample_model_parameters(
    trial: optuna.Trial,
    optimization_config: dict[str, Any],
) -> dict[str, Any]:
    """Sample one model parameter set from the configured search space."""
    search_space = optimization_config["search_space"]
    return {
        parameter_name: sample_optuna_parameter(
            trial,
            parameter_name,
            parameter_config,
        )
        for parameter_name, parameter_config in search_space.items()
    }


def sample_optuna_parameter(
    trial: optuna.Trial,
    parameter_name: str,
    parameter_config: dict[str, Any],
) -> Any:
    """Sample one configured Optuna parameter."""
    parameter_type = str(parameter_config["type"])
    if parameter_type == "int":
        return trial.suggest_int(
            parameter_name,
            int(parameter_config["low"]),
            int(parameter_config["high"]),
            step=int(parameter_config.get("step", 1)),
        )
    if parameter_type == "float":
        step = parameter_config.get("step")
        log = bool(parameter_config.get("log", False))
        if step is None:
            return trial.suggest_float(
                parameter_name,
                float(parameter_config["low"]),
                float(parameter_config["high"]),
                log=log,
            )
        return trial.suggest_float(
            parameter_name,
            float(parameter_config["low"]),
            float(parameter_config["high"]),
            step=float(step),
            log=log,
        )
    if parameter_type == "categorical":
        choices = parameter_config.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{parameter_name} categorical choices must not be empty.")
        return trial.suggest_categorical(parameter_name, choices)
    raise ValueError(f"Unsupported search space type for {parameter_name}: {parameter_type}")


def resolve_random_forest_parameters(
    modeling_parameters: dict[str, Any],
    selected_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge base and selected RandomForest parameters."""
    return resolve_model_parameters(
        modeling_parameters,
        "random_forest",
        selected_params=selected_params,
    )


def resolve_model_parameters(
    modeling_parameters: dict[str, Any],
    model_name: str,
    selected_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge base and selected parameters for one tuned model family."""
    validate_tuned_model_name(model_name)
    parameters = dict(modeling_parameters[model_name])
    if selected_params:
        parameters.update(selected_params)
    optimization_config = modeling_parameters.get(optimization_config_name(model_name), {})
    if "random_state" in parameters:
        parameters["random_state"] = int(
            optimization_config.get("random_state", parameters.get("random_state", 73))
        )
    if "n_jobs" in parameters:
        parameters["n_jobs"] = int(parameters.get("n_jobs", -1))
    if (
        model_name in BOOTSTRAP_MAX_SAMPLE_MODELS
        and not bool(parameters.get("bootstrap", True))
    ):
        parameters["max_samples"] = None
    validate_model_parameters(model_name, parameters)
    return parameters


def validate_random_forest_parameters(parameters: dict[str, Any]) -> None:
    """Validate resolved RandomForest parameters."""
    validate_model_parameters("random_forest", parameters)


def validate_model_parameters(model_name: str, parameters: dict[str, Any]) -> None:
    """Validate resolved model-family parameters."""
    validate_tuned_model_name(model_name)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"{model_name} parameters must be a non-empty mapping.")
    if "random_state" in parameters and int(parameters["random_state"]) != 73:
        raise ValueError(f"{model_name} random_state must remain fixed at 73.")
    if "n_jobs" in parameters and int(parameters["n_jobs"]) != -1:
        raise ValueError(f"{model_name} n_jobs must remain fixed at -1.")
    if model_name in N_ESTIMATOR_MODELS and "n_estimators" not in parameters:
        raise ValueError(f"{model_name} parameters must include n_estimators.")
    if model_name == "hist_gradient_boosting" and "max_iter" not in parameters:
        raise ValueError("hist_gradient_boosting parameters must include max_iter.")
    max_samples = parameters.get("max_samples")
    if (
        model_name in BOOTSTRAP_MAX_SAMPLE_MODELS
        and max_samples is not None
        and not bool(parameters.get("bootstrap", True))
    ):
        raise ValueError(f"{model_name} max_samples requires bootstrap=True.")


def with_model_parameters(
    modeling_parameters: dict[str, Any],
    model_name: str,
    model_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return modeling parameters with one model family's parameters replaced."""
    updated_parameters = dict(modeling_parameters)
    updated_parameters[model_name] = dict(model_parameters)
    return updated_parameters


def build_optuna_trial_summary_frame(study: optuna.Study) -> pd.DataFrame:
    """Build one-row-per-trial optimization metrics."""
    rows: list[dict[str, Any]] = []
    best_trial_number = study.best_trial.number
    for trial in study.trials:
        model_params = dict(trial.user_attrs["model_params"])
        cv_metrics = dict(trial.user_attrs["cv_metrics"])
        rows.append(
            {
                "trial_number": trial.number,
                "objective_value": float(trial.value),
                "is_best": trial.number == best_trial_number,
                **prefix_parameters(model_params),
                **cv_metrics,
            }
        )
    return pd.DataFrame(rows)


def build_disabled_optimization_artifacts(
    best_params: dict[str, Any],
    cv_result: modeling_evaluation.CrossValidationResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build optimization artifacts when tuning is disabled."""
    trial_frame = pd.DataFrame(
        [
            {
                "trial_number": 0,
                "objective_value": float(
                    cv_result.cv_metrics.loc[0, PRIMARY_DEVELOPMENT_METRIC]
                ),
                "is_best": True,
                **prefix_parameters(best_params),
                **cv_result.cv_metrics.iloc[0].to_dict(),
            }
        ]
    )
    trial_fold_frame = add_trial_context_to_fold_metrics(
        cv_result.cv_fold_metrics,
        trial_number=0,
        parameters=best_params,
    )
    return trial_frame, trial_fold_frame


def add_trial_context_to_fold_metrics(
    cv_fold_metrics: pd.DataFrame,
    trial_number: int,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Add trial metadata to per-fold CV metrics."""
    trial_fold_metrics = cv_fold_metrics.copy()
    trial_fold_metrics.insert(0, "trial_number", trial_number)
    for parameter_name, parameter_value in prefix_parameters(parameters).items():
        trial_fold_metrics[parameter_name] = parameter_value
    return trial_fold_metrics


def prefix_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Prefix model parameters for trial artifact columns."""
    return {
        f"{OPTUNA_TRIAL_PARAM_PREFIX}{parameter_name}": parameter_value
        for parameter_name, parameter_value in sorted(parameters.items())
    }
