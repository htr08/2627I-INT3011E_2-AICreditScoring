"""Thiết lập MLflow dùng chung cho script, notebook và test.

Ví dụ:
    from src.tracking import setup_mlflow
    import mlflow

    setup_mlflow()
    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_params({...})
        mlflow.log_metric("auc_cv", 0.76)
"""

import os

import mlflow

from src.config import PROJECT_ROOT, load_config


def get_tracking_uri(config: dict | None = None) -> str:
    """Ưu tiên biến môi trường MLFLOW_TRACKING_URI, nếu không có thì dùng thư mục trong config."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    config = config or load_config()
    tracking_dir = PROJECT_ROOT / config["mlflow"]["tracking_dir"]
    return tracking_dir.as_uri()


def setup_mlflow(experiment_name: str | None = None, config: dict | None = None) -> str:
    """Đặt tracking URI và experiment; trả về experiment_id."""
    config = config or load_config()
    mlflow.set_tracking_uri(get_tracking_uri(config))
    name = experiment_name or config["mlflow"]["experiment_name"]
    experiment = mlflow.set_experiment(name)
    return experiment.experiment_id
