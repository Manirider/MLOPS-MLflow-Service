from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
class TrainRequest(BaseModel):
    learning_rate: float = Field(
        default=0.001,
        ge=0.00001,
        le=1.0,
        description="Learning rate for optimizer"
    )
    epochs: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=64,
        ge=8,
        le=512,
        description="Batch size for training"
    )
    hidden_size: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Size of hidden layer"
    )
    dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Dropout/regularization rate"
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="MLflow experiment name (uses default if not specified)"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Optional name for this training run"
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "learning_rate": 0.001,
                    "epochs": 10,
                    "batch_size": 64,
                    "hidden_size": 128,
                    "dropout": 0.2,
                    "experiment_name": "MNIST_Experiments",
                    "run_name": "my_training_run"
                }
            ]
        }
    }
class TrainResponse(BaseModel):
    message: str = Field(description="Status message")
    job_id: str = Field(description="Internal Training Job ID")
    run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    experiment_name: str = Field(description="MLflow experiment name")
    status: TrainingStatus = Field(description="Training job status")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Training initiated successfully",
                    "run_id": "abc123def456",
                    "experiment_name": "MNIST_Experiments",
                    "status": "running"
                }
            ]
        }
    }
class TrainStatusResponse(BaseModel):
    run_id: str = Field(description="MLflow run ID")
    status: TrainingStatus = Field(description="Current training status")
    metrics: Optional[dict] = Field(default=None, description="Training metrics if available")
    error: Optional[str] = Field(default=None, description="Error message if failed")
