import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.models import (
    RegisteredModel,
    ModelsResponse,
    ModelVersionInfo,
    TransitionStageRequest,
    TransitionStageResponse,
)
from app.services.mlflow_service import get_mlflow_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])

@router.get(
    "",
    response_model=ModelsResponse,
    summary="List all registered models",
    description="Get a list of all models in the MLflow Model Registry",
)
async def list_models():
    try:
        mlflow_service = get_mlflow_service()
        models = mlflow_service.list_registered_models()

        registered_models = []
        for model in models:
            versions = [
                ModelVersionInfo(
                    version=v["version"],
                    stage=v["stage"],
                    run_id=v["run_id"],
                    status=v["status"],
                    creation_timestamp=v.get("creation_timestamp"),
                )
                for v in model.get("versions", [])
            ]

            registered_models.append(RegisteredModel(
                name=model["name"],
                description=model.get("description"),
                latest_version=model.get("latest_version"),
                latest_stage=model.get("latest_stage"),
                creation_timestamp=model.get("creation_timestamp"),
                last_updated_timestamp=model.get("last_updated_timestamp"),
                versions=versions,
            ))

        return ModelsResponse(
            models=registered_models,
            total_count=len(registered_models),
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
        )

@router.get(
    "/{model_name}",
    response_model=RegisteredModel,
    summary="Get model details",
    description="Get detailed information about a specific registered model",
)
async def get_model(model_name: str):
    try:
        mlflow_service = get_mlflow_service()
        models = mlflow_service.list_registered_models()

        for model in models:
            if model["name"] == model_name:
                versions = [
                    ModelVersionInfo(
                        version=v["version"],
                        stage=v["stage"],
                        run_id=v["run_id"],
                        status=v["status"],
                        creation_timestamp=v.get("creation_timestamp"),
                    )
                    for v in model.get("versions", [])
                ]

                return RegisteredModel(
                    name=model["name"],
                    description=model.get("description"),
                    latest_version=model.get("latest_version"),
                    latest_stage=model.get("latest_stage"),
                    creation_timestamp=model.get("creation_timestamp"),
                    last_updated_timestamp=model.get("last_updated_timestamp"),
                    versions=versions,
                )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

@router.post(
    "/transition",
    response_model=TransitionStageResponse,
    summary="Transition model stage",
    description="Transition a model version to a new stage",
)
async def transition_stage(request: TransitionStageRequest):
    try:
        mlflow_service = get_mlflow_service()

        mlflow_service.transition_model_stage(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage.value,
            archive_existing=request.archive_existing,
        )

        logger.info(
            f"Transitioned {request.model_name} v{request.version} to {request.stage.value}"
        )

        return TransitionStageResponse(
            message=f"Successfully transitioned to {request.stage.value}",
            model_name=request.model_name,
            version=request.version,
            stage=request.stage.value,
        )

    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transition model stage: {str(e)}"
        )
