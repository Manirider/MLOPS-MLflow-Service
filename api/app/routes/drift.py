import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.drift_service import get_drift_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/drift", tags=["Drift Detection"])

@router.get(
    "/status",
    summary="Get model drift status",
    description="Calculate and return current drift metrics using PSI"
)
async def get_drift_status(
    model_name: str = Query(default="MNISTClassifier", description="Model name to analyze"),
    baseline_hours: int = Query(default=168, description="Hours for baseline window (default: 1 week)"),
    current_hours: int = Query(default=24, description="Hours for current window (default: 24h)")
):
    drift_service = get_drift_service()

    try:
        status = drift_service.get_drift_status(
            model_name=model_name,
            baseline_hours=baseline_hours,
            current_hours=current_hours
        )
        return status
    except Exception as e:
        logger.error(f"Drift status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/stats",
    summary="Get prediction statistics",
    description="Return prediction statistics for monitoring"
)
async def get_prediction_stats(
    model_name: str = Query(default="MNISTClassifier", description="Model name to analyze"),
    hours: int = Query(default=24, description="Time window in hours")
):
    drift_service = get_drift_service()

    try:
        stats = drift_service.get_prediction_stats(
            model_name=model_name,
            hours=hours
        )
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/distribution",
    summary="Get prediction distribution",
    description="Return class distribution for predictions"
)
async def get_distribution(
    model_name: str = Query(default="MNISTClassifier", description="Model name to analyze"),
    hours: int = Query(default=24, description="Time window in hours")
):
    drift_service = get_drift_service()

    try:
        distribution = drift_service.get_prediction_distribution(
            model_name=model_name,
            hours=hours
        )
        return {
            "model_name": model_name,
            "time_window_hours": hours,
            "distribution": distribution,
            "total_classes": len(distribution)
        }
    except Exception as e:
        logger.error(f"Distribution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
