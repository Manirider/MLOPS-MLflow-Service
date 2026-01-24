from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class ModelVersionInfo(BaseModel):

    version: str = Field(description="Model version number")
    stage: str = Field(description="Current stage")
    run_id: str = Field(description="Training run ID")
    status: str = Field(description="Model status")
    creation_timestamp: Optional[int] = Field(default=None, description="Creation timestamp")
    description: Optional[str] = Field(default=None, description="Version description")

class RegisteredModel(BaseModel):

    name: str = Field(description="Model name")
    description: Optional[str] = Field(default=None, description="Model description")
    latest_version: Optional[str] = Field(default=None, description="Latest version number")
    latest_stage: Optional[str] = Field(default=None, description="Latest version stage")
    creation_timestamp: Optional[int] = Field(default=None, description="Creation timestamp")
    last_updated_timestamp: Optional[int] = Field(default=None, description="Last update timestamp")
    versions: List[ModelVersionInfo] = Field(default_factory=list, description="All versions")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "MNISTClassifier",
                    "description": "MNIST digit classifier",
                    "latest_version": "3",
                    "latest_stage": "Production",
                    "versions": [
                        {
                            "version": "3",
                            "stage": "Production",
                            "run_id": "abc123",
                            "status": "READY"
                        }
                    ]
                }
            ]
        }
    }

class ModelsResponse(BaseModel):

    models: List[RegisteredModel] = Field(description="List of registered models")
    total_count: int = Field(description="Total number of models")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "models": [
                        {
                            "name": "MNISTClassifier",
                            "latest_version": "3",
                            "latest_stage": "Production"
                        }
                    ],
                    "total_count": 1
                }
            ]
        }
    }

class TransitionStageRequest(BaseModel):

    model_name: str = Field(description="Registered model name")
    version: str = Field(description="Version to transition")
    stage: ModelStage = Field(description="Target stage")
    archive_existing: bool = Field(default=True, description="Archive existing models in stage")

class TransitionStageResponse(BaseModel):

    message: str = Field(description="Status message")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    stage: str = Field(description="New stage")
