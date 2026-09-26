import mlflow

from src.config import load_config
from src.tracking import get_tracking_uri, setup_mlflow


def test_load_config_has_required_keys():
    config = load_config()
    assert config["random_state"] == 42
    assert config["mlflow"]["experiment_name"]
    assert config["mlflow"]["tracking_dir"]


def test_env_var_overrides_tracking_uri(monkeypatch, tmp_path):
    uri = tmp_path.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    assert get_tracking_uri() == uri


def test_log_nested_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    experiment_id = setup_mlflow("pytest_experiment")

    with mlflow.start_run(run_name="parent") as parent:
        mlflow.log_param("model", "logreg")
        with mlflow.start_run(run_name="trial_0", nested=True):
            mlflow.log_metric("auc", 0.75)

    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    assert len(runs) == 2
    child = runs[runs["tags.mlflow.parentRunId"] == parent.info.run_id]
    assert child["metrics.auc"].iloc[0] == 0.75
