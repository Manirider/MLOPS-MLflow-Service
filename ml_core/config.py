import os
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class MLConfig:
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("EXPERIMENT_NAME", "MNIST_Experiments")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "MNISTClassifier")
    )
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_LEARNING_RATE", "0.001"))
    )
    epochs: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_EPOCHS", "10"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_BATCH_SIZE", "64"))
    )
    hidden_size: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_HIDDEN_SIZE", "128"))
    )
    dropout: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_DROPOUT", "0.2"))
    )
    random_seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )
    data_dir: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "./data")
    )
    artifact_dir: str = field(
        default_factory=lambda: os.getenv("ARTIFACT_DIR", "./mlruns")
    )
def get_config(**overrides) -> MLConfig:
    config = MLConfig()
    for key, value in overrides.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config
