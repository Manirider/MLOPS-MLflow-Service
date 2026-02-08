import itertools
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [10, 20, None]
}
EXPERIMENT = "MNIST_Experiments"
MODEL_NAME = "MNISTClassifier"
def run():
    for n, d in itertools.product(
        PARAM_GRID["n_estimators"],
        PARAM_GRID["max_depth"]
    ):
        subprocess.run([
            "python",
            "ml_core/training/train.py",
            "--n_estimators", str(n),
            "--max_depth", str(d),
            "--experiment", EXPERIMENT
        ])
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        order_by=["metrics.accuracy DESC"]
    )
    best_run = runs.iloc[0]
    run_id = best_run.run_id
    client = MlflowClient()
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging"
    )
if __name__ == "__main__":
    run()
