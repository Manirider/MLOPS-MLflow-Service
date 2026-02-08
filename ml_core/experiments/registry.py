import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_core.config import get_config
def get_mlflow_client() -> MlflowClient:
    config = get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    return MlflowClient()
def register_model_from_run(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> str:
    client = get_mlflow_client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            model_name,
            description=f"MNIST Classifier model trained with MLOps platform"
        )
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Model from run {run_id}"
    )
    return model_version.version
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> None:
    client = get_mlflow_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    print(f"Transitioned {model_name} v{version} to {stage}")
def get_latest_model_version(
    model_name: str,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    client = get_mlflow_client()
    try:
        if stage:
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = client.get_latest_versions(model_name)
        if not versions:
            return None
        latest = versions[0]
        return {
            "version": latest.version,
            "stage": latest.current_stage,
            "run_id": latest.run_id,
            "source": latest.source,
            "status": latest.status,
            "creation_timestamp": latest.creation_timestamp,
        }
    except MlflowException:
        return None
def get_all_registered_models() -> List[Dict[str, Any]]:
    client = get_mlflow_client()
    models = []
    for rm in client.search_registered_models():
        latest_versions = client.get_latest_versions(rm.name)
        model_info = {
            "name": rm.name,
            "description": rm.description,
            "creation_timestamp": rm.creation_timestamp,
            "last_updated_timestamp": rm.last_updated_timestamp,
            "versions": [],
        }
        for version in latest_versions:
            model_info["versions"].append({
                "version": version.version,
                "stage": version.current_stage,
                "run_id": version.run_id,
                "status": version.status,
            })
        models.append(model_info)
    return models
def get_production_model_uri(model_name: str) -> Optional[str]:
    latest = get_latest_model_version(model_name, stage="Production")
    if latest:
        return f"models:/{model_name}/Production"
    return None
def delete_model_version(model_name: str, version: str) -> None:
    client = get_mlflow_client()
    client.delete_model_version(model_name, version)
def update_model_description(
    model_name: str,
    version: str,
    description: str,
) -> None:
    client = get_mlflow_client()
    client.update_model_version(
        name=model_name,
        version=version,
        description=description,
    )
