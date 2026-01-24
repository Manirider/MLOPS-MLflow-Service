import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.schemas.train import TrainRequest, TrainResponse, TrainingStatus
from app.services.training_service import get_training_service
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["Training"])

@router.post(
    "",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a training run",
    description="Trigger an asynchronous training job with specified hyperparameters",
)
async def start_training(request: TrainRequest):
    settings = get_settings()
    training_service = get_training_service()

    experiment_name = request.experiment_name or settings.experiment_name

    try:
        job = await training_service.start_training(
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            batch_size=request.batch_size,
            hidden_size=request.hidden_size,
            dropout=request.dropout,
            experiment_name=experiment_name,
            run_name=request.run_name,
        )

        logger.info(f"Training job started: {job.job_id}")

        return TrainResponse(
            message="Training initiated successfully",
            job_id=job.job_id,
            run_id=job.run_id,
            experiment_name=experiment_name,
            status=TrainingStatus.RUNNING,
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )

@router.get(
    "/status/{job_id}",
    summary="Get training job status",
    description="Check the status of a running or completed training job",
)
async def get_training_status(job_id: str):
    training_service = get_training_service()

    job = training_service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{job_id}' not found"
        )

    return {
        "job_id": job.job_id,
        "run_id": job.run_id,
        "experiment_name": job.experiment_name,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "params": job.params,
    }

@router.get(
    "/jobs",
    summary="List all training jobs",
    description="Get a list of all training jobs",
)
async def list_training_jobs():
    training_service = get_training_service()

    jobs = training_service.list_jobs()

    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "run_id": job.run_id,
                "experiment_name": job.experiment_name,
                "status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
            }
            for job in jobs
        ],
        "total_count": len(jobs),
    }
