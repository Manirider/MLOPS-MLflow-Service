import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

class MLflowService:

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[MlflowClient] = None
        self._initialize_mlflow()

    def _initialize_mlflow(self):
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {self.settings.mlflow_tracking_uri}")

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient(self.settings.mlflow_tracking_uri)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_or_create_experiment(self, name: str) -> str:
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        return self.client.create_experiment(name)

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = []

        for exp in self.client.search_experiments():
            exp_info = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }

            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1,
            )
            exp_info["total_runs"] = len(self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1000,
            ))

            if runs:
                best_runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["metrics.accuracy DESC"],
                    max_results=1,
                )
                if best_runs:
                    best = best_runs[0]
                    exp_info["best_run"] = {
                        "run_id": best.info.run_id,
                        "run_name": best.info.run_name,
                        "status": best.info.status,
                        "start_time": best.info.start_time,
                        "end_time": best.info.end_time,
                        "metrics": dict(best.data.metrics),
                        "params": dict(best.data.params),
                    }

            experiments.append(exp_info)

        return experiments

    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        exp = self.client.get_experiment_by_name(name)
        if not exp:
            return None

        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
        }

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        runs = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=order_by or [],
            max_results=max_results,
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
            }
            for run in runs
        ]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            run = self.client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "artifact_uri": run.info.artifact_uri,
            }
        except MlflowException:
            return None

    def list_registered_models(self) -> List[Dict[str, Any]]:
        models = []

        for rm in self.client.search_registered_models():
            versions = self.client.get_latest_versions(rm.name)

            latest_version = None
            latest_stage = None
            if versions:
                latest_version = versions[0].version
                latest_stage = versions[0].current_stage

            model_info = {
                "name": rm.name,
                "description": rm.description,
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
                "latest_version": latest_version,
                "latest_stage": latest_stage,
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "status": v.status,
                        "creation_timestamp": v.creation_timestamp,
                    }
                    for v in versions
                ],
            }
            models.append(model_info)

        return models

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                return None

            v = versions[0]
            return {
                "name": model_name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "source": v.source,
                "status": v.status,
            }
        except MlflowException:
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

@lru_cache()
def get_mlflow_service() -> MLflowService:
    return MLflowService()
