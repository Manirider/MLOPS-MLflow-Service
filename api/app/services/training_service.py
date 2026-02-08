import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from celery.result import AsyncResult
from app.config import get_settings
from app.schemas.train import TrainingStatus
from app.tasks import train_model_task
from app.worker import celery_app
logger = logging.getLogger(__name__)
@dataclass
class TrainingJob:
    job_id: str
    run_id: Optional[str] = None
    experiment_name: str = ""
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
class TrainingService:
    def __init__(self):
        self.settings = get_settings()
        self._jobs: Dict[str, TrainingJob] = {}
    async def start_training(
        self,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        hidden_size: int,
        dropout: float,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> TrainingJob:
        exp_name = experiment_name or self.settings.experiment_name
        params = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "run_name": run_name,
            "random_seed": self.settings.random_seed
        }
        task = train_model_task.delay(
            metrics={},
            params=params,
            experiment_name=exp_name,
            run_name=run_name
        )
        job_id = task.id
        job = TrainingJob(
            job_id=job_id,
            experiment_name=exp_name,
            status=TrainingStatus.PENDING,
            params=params,
            started_at=datetime.utcnow()
        )
        self._jobs[job_id] = job
        logger.info(f"Dispatched training job {job_id} to Celery")
        return job
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        task_result = AsyncResult(job_id, app=celery_app)
        if task_result.state == 'SUCCESS':
            job.status = TrainingStatus.COMPLETED
            result_data = task_result.result
            if isinstance(result_data, dict):
                job.run_id = result_data.get("run_id")
            job.completed_at = datetime.utcnow()
        elif task_result.state == 'FAILURE':
            job.status = TrainingStatus.FAILED
            job.error = str(task_result.result)
            job.completed_at = datetime.utcnow()
        elif task_result.state in ['STARTED', 'RETRY']:
            job.status = TrainingStatus.RUNNING
        return job
    def list_jobs(self) -> list:
        for job_id in self._jobs:
            self.get_job(job_id)
        return list(self._jobs.values())
_training_service: Optional[TrainingService] = None
def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
