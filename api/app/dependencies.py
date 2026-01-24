from typing import Generator

from app.services.mlflow_service import MLflowService, get_mlflow_service
from app.services.training_service import TrainingService, get_training_service
from app.services.inference_service import InferenceService, get_inference_service
from app.config import Settings, get_settings

def get_settings_dependency() -> Settings:
    return get_settings()

def get_mlflow_service_dependency() -> MLflowService:
    return get_mlflow_service()

def get_training_service_dependency() -> TrainingService:
    return get_training_service()

def get_inference_service_dependency() -> InferenceService:
    return get_inference_service()
