import logging
import json
import hashlib
import redis
from fastapi import APIRouter, HTTPException, status

from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from app.services.inference_service import get_inference_service
from app.services.drift_service import get_drift_service
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.post(
    "",
    response_model=PredictResponse,
    summary="Get prediction",
    description="Get prediction for an MNIST image using the Production model",
)
async def predict(request: PredictRequest):
    settings = get_settings()

    try:
        r = redis.Redis(host=settings.redis_host,
                        port=settings.redis_port, db=0, decode_responses=True)

        input_str = json.dumps(request.image)
        cache_key = hashlib.md5(f"predict:{input_str}".encode()).hexdigest()

        cached_result = r.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key: {cache_key}")
            return PredictResponse(**json.loads(cached_result))
    except Exception as e:
        logger.warning(f"Redis cache error: {e}")
        r = None

    try:
        inference_service = get_inference_service()

        result = inference_service.predict(
            image_data=request.image,
            model_name=settings.model_name,
            stage="Production",
        )

        response = PredictResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            model_stage=result["model_stage"],
        )

        if r:
            try:
                r.setex(
                    cache_key,
                    settings.redis_ttl,
                    json.dumps(response.dict())
                )
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

        try:
            drift_service = get_drift_service()
            drift_service.log_prediction(
                model_name=result["model_name"],
                input_features=request.image[:10],
                prediction=result["prediction"],
                confidence=result["confidence"],
                model_version=result.get("model_version")
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction for drift: {e}")

        return response

    except ValueError as e:
        logger.warning(f"Prediction failed - no production model: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Get batch predictions",
    description="Get predictions for multiple MNIST images",
)
async def predict_batch(request: BatchPredictRequest):
    settings = get_settings()

    try:
        inference_service = get_inference_service()

        result = inference_service.predict_batch(
            images=request.images,
            model_name=settings.model_name,
            stage="Production",
        )

        return BatchPredictResponse(
            predictions=result["predictions"],
            confidences=result["confidences"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            batch_size=result["batch_size"],
        )

    except ValueError as e:
        logger.warning(f"Batch prediction failed - no production model: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.post(
    "/staging",
    response_model=PredictResponse,
    summary="Get prediction from Staging model",
    description="Get prediction using the Staging-staged model (for testing)",
)
async def predict_staging(request: PredictRequest):
    settings = get_settings()

    try:
        inference_service = get_inference_service()

        result = inference_service.predict(
            image_data=request.image,
            model_name=settings.model_name,
            stage="Staging",
        )

        return PredictResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            model_stage=result["model_stage"],
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Staging prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.delete(
    "/cache",
    summary="Clear model cache",
    description="Clear the inference service model cache",
)
async def clear_cache():
    try:
        inference_service = get_inference_service()
        inference_service.clear_cache()

        return {"message": "Model cache cleared successfully"}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )
