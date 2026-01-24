import logging
from fastapi import APIRouter, HTTPException, status, Request
from typing import Optional

from app.schemas.experiments import (
    ExperimentSummary,
    ExperimentsResponse,
    RunSummary,
    ExperimentDetailResponse,
)
from app.services.mlflow_service import get_mlflow_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["Experiments"])

from app.middleware.security import limiter

@router.get(
    "",
    response_model=ExperimentsResponse,
    summary="List all experiments",
    description="Get a list of all MLflow experiments with their best runs",
)
@limiter.limit("100/minute")
async def list_experiments(request: Request):
    try:
        mlflow_service = get_mlflow_service()
        experiments = mlflow_service.list_experiments()

        experiment_summaries = []
        for exp in experiments:
            best_run = None
            if exp.get("best_run"):
                br = exp["best_run"]
                best_run = RunSummary(
                    run_id=br["run_id"],
                    run_name=br.get("run_name"),
                    status=br["status"],
                    start_time=br.get("start_time"),
                    end_time=br.get("end_time"),
                    metrics=br.get("metrics", {}),
                    params=br.get("params", {}),
                )

            summary = ExperimentSummary(
                experiment_id=exp["experiment_id"],
                name=exp["name"],
                artifact_location=exp.get("artifact_location"),
                lifecycle_stage=exp.get("lifecycle_stage", "active"),
                total_runs=exp.get("total_runs", 0),
                best_run=best_run,
            )
            experiment_summaries.append(summary)

        return ExperimentsResponse(
            experiments=experiment_summaries,
            total_count=len(experiment_summaries),
        )

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiments: {str(e)}"
        )

@router.get(
    "/{experiment_name}",
    response_model=ExperimentDetailResponse,
    summary="Get experiment details",
    description="Get detailed information about a specific experiment",
)
async def get_experiment(experiment_name: str, max_runs: int = 100):
    try:
        mlflow_service = get_mlflow_service()

        exp = mlflow_service.get_experiment_by_name(experiment_name)

        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{experiment_name}' not found"
            )

        runs = mlflow_service.search_runs(
            experiment_ids=[exp["experiment_id"]],
            order_by=["start_time DESC"],
            max_results=max_runs,
        )

        run_summaries = [
            RunSummary(
                run_id=run["run_id"],
                run_name=run.get("run_name"),
                status=run["status"],
                start_time=run.get("start_time"),
                end_time=run.get("end_time"),
                metrics=run.get("metrics", {}),
                params=run.get("params", {}),
            )
            for run in runs
        ]

        return ExperimentDetailResponse(
            experiment_id=exp["experiment_id"],
            name=exp["name"],
            artifact_location=exp.get("artifact_location"),
            lifecycle_stage=exp.get("lifecycle_stage", "active"),
            runs=run_summaries,
            total_runs=len(run_summaries),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment '{experiment_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiment: {str(e)}"
        )

@router.get(
    "/{experiment_name}/runs/{run_id}",
    summary="Get run details",
    description="Get detailed information about a specific run",
)
async def get_run(experiment_name: str, run_id: str):
    try:
        mlflow_service = get_mlflow_service()

        run = mlflow_service.get_run(run_id)

        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )

        return run

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run '{run_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve run: {str(e)}"
        )
