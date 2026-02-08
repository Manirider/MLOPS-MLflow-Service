import argparse
import os
import sys
import itertools
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_core.config import get_config
from ml_core.training.train import train_model
from ml_core.experiments.registry import (
    register_model_from_run,
    transition_model_stage,
    get_latest_model_version,
)
def generate_hyperparameter_grid() -> List[Dict[str, Any]]:
    learning_rates = [0.0001, 0.001, 0.01]
    hidden_sizes = [64, 128, 256]
    batch_sizes = [32, 64, 128]
    dropouts = [0.1, 0.2, 0.3]
    combinations = list(itertools.product(
        learning_rates, hidden_sizes, batch_sizes, dropouts
    ))
    return [
        {
            "learning_rate": lr,
            "hidden_size": hs,
            "batch_size": bs,
            "dropout": dr,
        }
        for lr, hs, bs, dr in combinations
    ]
def generate_random_hyperparameters(n_samples: int = 10) -> List[Dict[str, Any]]:
    configs = []
    for i in range(n_samples):
        config = {
            "learning_rate": random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01]),
            "hidden_size": random.choice([64, 96, 128, 192, 256]),
            "batch_size": random.choice([32, 64, 128]),
            "dropout": random.uniform(0.1, 0.4),
            "epochs": random.choice([5, 10, 15]),
        }
        configs.append(config)
    return configs
def run_experiments(
    num_runs: int = 10,
    experiment_name: str = "MNIST_Experiments",
    epochs: int = 5,
    search_strategy: str = "random",
    random_seed: int = 42,
) -> List[str]:
    random.seed(random_seed)
    if search_strategy == "grid":
        all_configs = generate_hyperparameter_grid()
        configs = random.sample(all_configs, min(num_runs, len(all_configs)))
    else:
        configs = generate_random_hyperparameters(num_runs)
    run_ids = []
    print(f"\n{'='*60}")
    print(f"Running {len(configs)} experiments")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}\n")
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running experiment with config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        run_name = f"run_{i+1:03d}_lr{config['learning_rate']}_hs{config['hidden_size']}"
        run_id = train_model(
            learning_rate=config["learning_rate"],
            epochs=config.get("epochs", epochs),
            batch_size=config["batch_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            random_seed=random_seed + i,
            experiment_name=experiment_name,
            run_name=run_name,
        )
        run_ids.append(run_id)
        print(f"  Completed: {run_id}")
    return run_ids
def find_best_run(
    experiment_name: str,
    metric: str = "accuracy",
    ascending: bool = False,
) -> Optional[Dict[str, Any]]:
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        print("No runs found in experiment")
        return None
    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "experiment_id": experiment.experiment_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifact_uri": best_run.info.artifact_uri,
    }
def run_and_register_best(
    num_runs: int = 10,
    experiment_name: str = "MNIST_Experiments",
    model_name: str = "MNISTClassifier",
    epochs: int = 5,
    metric: str = "accuracy",
    stage: str = "Staging",
) -> Dict[str, Any]:
    run_ids = run_experiments(
        num_runs=num_runs,
        experiment_name=experiment_name,
        epochs=epochs,
    )
    print(f"\n{'='*60}")
    print("Finding best run...")
    print(f"{'='*60}")
    best_run = find_best_run(experiment_name, metric=metric)
    if best_run is None:
        return {"error": "No runs found"}
    print(f"\nBest run: {best_run['run_id']}")
    print(f"Metrics: {best_run['metrics']}")
    print(f"\nRegistering model as '{model_name}'...")
    version = register_model_from_run(
        run_id=best_run["run_id"],
        model_name=model_name,
    )
    print(f"Registered version: {version}")
    print(f"\nTransitioning to '{stage}' stage...")
    transition_model_stage(model_name, version, stage)
    return {
        "run_ids": run_ids,
        "best_run": best_run,
        "model_name": model_name,
        "model_version": version,
        "stage": stage,
    }
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter experiments and register best model"
    )
    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=10,
        help="Number of experiments to run"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.getenv("EXPERIMENT_NAME", "MNIST_Experiments"),
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "MNISTClassifier"),
        help="Name for registered model"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Training epochs per run"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Metric for best model selection"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        choices=["None", "Staging", "Production", "Archived"],
        help="Stage to transition best model to"
    )
    parser.add_argument(
        "--register-best",
        action="store_true",
        help="Register the best model after runs"
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Hyperparameter search strategy"
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    if args.register_best:
        results = run_and_register_best(
            num_runs=args.num_runs,
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            epochs=args.epochs,
            metric=args.metric,
            stage=args.stage,
        )
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Total runs: {len(results.get('run_ids', []))}")
        print(f"Best run ID: {results.get('best_run', {}).get('run_id')}")
        print(f"Best {args.metric}: {results.get('best_run', {}).get('metrics', {}).get(args.metric)}")
        print(f"Model: {results.get('model_name')} v{results.get('model_version')}")
        print(f"Stage: {results.get('stage')}")
    else:
        run_ids = run_experiments(
            num_runs=args.num_runs,
            experiment_name=args.experiment_name,
            epochs=args.epochs,
            search_strategy=args.search_strategy,
        )
        print(f"\n{'='*60}")
        print(f"Completed {len(run_ids)} runs")
        print(f"{'='*60}")
        best = find_best_run(args.experiment_name, metric=args.metric)
        if best:
            print(f"\nBest run: {best['run_id']}")
            print(f"Accuracy: {best['metrics'].get('accuracy', 'N/A')}")
