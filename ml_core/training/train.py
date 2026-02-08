from ml_core.training.artifacts import save_training_artifacts
from ml_core.training.evaluate import evaluate_model, generate_classification_report
from ml_core.models.mnist_cnn import MNISTClassifier
from ml_core.config import get_config
import argparse
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
def set_seeds(seed: int):
    np.random.seed(seed)
def load_mnist_data(data_dir: str = "./data"):
    os.makedirs(data_dir, exist_ok=True)
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1,
                         as_frame=False, data_home=data_dir)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test
def train_model(
    learning_rate: float = 0.001,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    dropout: float = 0.2,
    random_seed: int = 42,
    experiment_name: str = "MNIST_Experiments",
    run_name: str = None,
) -> str:
    set_seeds(random_seed)
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    X_train, X_test, y_train, y_test = load_mnist_data(config.data_dir)
    model = MNISTClassifier(
        hidden_layer_sizes=(hidden_size, hidden_size // 2),
        learning_rate_init=learning_rate,
        max_iter=epochs,
        batch_size=batch_size,
        alpha=dropout,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("model_type", "MLPClassifier")
        mlflow.log_param("hidden_layer_sizes",
                         f"({hidden_size}, {hidden_size // 2})")
        print(f"\nTraining model with run_id: {run.info.run_id}")
        print(
            f"Parameters: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None,
        )
        try:
            print("ONNX export skipped: Model is sklearn pipeline, requires skl2onnx.")
        except Exception as e:
            print(f"ONNX export failed: {e}")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_artifacts(
                model, X_test, y_test,
                output_dir=tmpdir,
                run_id=run.info.run_id
            )
            for artifact_file in Path(tmpdir).glob("*"):
                mlflow.log_artifact(str(artifact_file))
        print(f"\nTraining complete. Run ID: {run.info.run_id}")
        print(f"View at: {config.mlflow_tracking_uri}")
        return run.info.run_id
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST classifier with MLflow tracking")
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=float(os.getenv("DEFAULT_LEARNING_RATE", "0.001")),
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=int(os.getenv("DEFAULT_EPOCHS", "10")),
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=int(os.getenv("DEFAULT_BATCH_SIZE", "64")),
        help="Batch size for training"
    )
    parser.add_argument(
        "--hidden-size", "-hs",
        type=int,
        default=int(os.getenv("DEFAULT_HIDDEN_SIZE", "128")),
        help="Size of hidden layer"
    )
    parser.add_argument(
        "--dropout", "-d",
        type=float,
        default=float(os.getenv("DEFAULT_DROPOUT", "0.2")),
        help="Dropout/regularization rate"
    )
    parser.add_argument(
        "--random-seed", "-s",
        type=int,
        default=int(os.getenv("RANDOM_SEED", "42")),
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=os.getenv("EXPERIMENT_NAME", "MNIST_Experiments"),
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run"
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    run_id = train_model(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        random_seed=args.random_seed,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"\nRun ID: {run_id}")
