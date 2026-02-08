from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
class RunSummary(BaseModel):
    run_id: str = Field(description="MLflow run ID")
    run_name: Optional[str] = Field(default=None, description="Run name if set")
    status: str = Field(description="Run status")
    start_time: Optional[int] = Field(default=None, description="Run start timestamp")
    end_time: Optional[int] = Field(default=None, description="Run end timestamp")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Run metrics")
    params: Dict[str, str] = Field(default_factory=dict, description="Run parameters")
class ExperimentSummary(BaseModel):
    experiment_id: str = Field(description="MLflow experiment ID")
    name: str = Field(description="Experiment name")
    artifact_location: Optional[str] = Field(default=None, description="Artifact storage location")
    lifecycle_stage: str = Field(default="active", description="Lifecycle stage")
    total_runs: int = Field(default=0, description="Total number of runs")
    best_run: Optional[RunSummary] = Field(default=None, description="Best performing run")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "experiment_id": "1",
                    "name": "MNIST_Experiments",
                    "artifact_location": "/mlruns/1",
                    "lifecycle_stage": "active",
                    "total_runs": 10,
                    "best_run": {
                        "run_id": "abc123",
                        "run_name": "best_run",
                        "status": "FINISHED",
                        "metrics": {"accuracy": 0.95},
                        "params": {"learning_rate": "0.001"}
                    }
                }
            ]
        }
    }
class ExperimentsResponse(BaseModel):
    experiments: List[ExperimentSummary] = Field(description="List of experiments")
    total_count: int = Field(description="Total number of experiments")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "experiments": [
                        {
                            "experiment_id": "1",
                            "name": "MNIST_Experiments",
                            "total_runs": 10
                        }
                    ],
                    "total_count": 1
                }
            ]
        }
    }
class ExperimentDetailResponse(BaseModel):
    experiment_id: str = Field(description="MLflow experiment ID")
    name: str = Field(description="Experiment name")
    artifact_location: Optional[str] = Field(default=None, description="Artifact storage location")
    lifecycle_stage: str = Field(default="active", description="Lifecycle stage")
    runs: List[RunSummary] = Field(default_factory=list, description="All runs in experiment")
    total_runs: int = Field(default=0, description="Total number of runs")
