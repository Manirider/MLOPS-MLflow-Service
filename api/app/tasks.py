import logging
from app.worker import celery_app
from ml_core.training.train import train_model
from app.services.mlflow_service import get_mlflow_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="train_model_task")
def train_model_task(self, metrics: dict, params: dict, experiment_name: str, run_name: str = None):
    logger.info(f"Starting training task: {self.request.id}")

    try:

        run_id = train_model(
            learning_rate=params.get("learning_rate"),
            epochs=params.get("epochs"),
            batch_size=params.get("batch_size"),
            hidden_size=params.get("hidden_size"),
            dropout=params.get("dropout"),
            random_seed=params.get("random_seed"),
            experiment_name=experiment_name,
            run_name=run_name
        )

        return {
            "status": "success",
            "run_id": run_id,
            "message": "Training completed successfully"
        }

    except Exception as e:
        logger.error(f"Training task failed: {str(e)}")

        return {
            "status": "failed",
            "error": str(e)
        }
