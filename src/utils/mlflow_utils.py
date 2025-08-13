import os
from typing import Any, Dict

import mlflow


def setup_mlflow(config: Dict[str, Any], experiment_name: str = "medical_image_translation"):
    """
    Sets up MLFlow tracking with the given configuration.

    Args:
        config: Configuration dictionary
        experiment_name: Name of the MLFlow experiment

    Returns:
        active_run: MLFlow active run object
    """
    # Set tracking URI if specified in config
    if "mlflow" in config and "tracking_uri" in config["mlflow"]:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    else:
        mlflow.set_tracking_uri("http://localhost:5000")

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Start the run
    run_name = config.get("run_name", None)
    active_run = mlflow.start_run(run_name=run_name)


    # Log tags if specified
    if "mlflow" in config and "tags" in config["mlflow"]:
        mlflow.set_tags(config["mlflow"]["tags"])

    return active_run

def log_config(config: Dict[str, Any], prefix: str = ""):
    """
    Recursively logs configuration parameters to MLFlow.

    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names in nested dictionaries
    """
    for key, value in config.items():
        param_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            log_config(value, f"{param_name}.")
        elif isinstance(value, (int, float, str, bool)):
            mlflow.log_param(param_name, value)
