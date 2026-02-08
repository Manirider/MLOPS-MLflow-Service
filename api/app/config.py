import os
from pydantic_settings import BaseSettings
from functools import lru_cache
class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "MLOps Platform API"
    api_version: str = "1.0.0"
    api_description: str = "Enterprise-grade MLOps experiment tracking and model registry"
    api_key: str = "mlops-secret-key-123"
    enable_rate_limit: bool = True
    mlflow_tracking_uri: str = "http://mlflow_server:5000"
    experiment_name: str = "MNIST_Experiments"
    model_name: str = "MNISTClassifier"
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_ttl: int = 3600
    postgres_db: str = "mlflow_db"
    postgres_user: str = "mlflow_user"
    postgres_password: str = "mlflow_password"
    default_learning_rate: float = 0.001
    default_epochs: int = 10
    default_batch_size: int = 64
    default_hidden_size: int = 128
    default_dropout: float = 0.2
    random_seed: int = 42
    data_dir: str = "/app/data"
    artifact_dir: str = "/mlruns"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
@lru_cache()
def get_settings() -> Settings:
    return Settings()
