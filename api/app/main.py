import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.middleware.security import limiter, SecurityMiddleware, get_api_key
from fastapi import Depends
from app.config import get_settings
from app.routes import (
    train_router,
    experiments_router,
    models_router,
    predict_router,
    drift_router,
)
from app.middleware.logging import LoggingMiddleware
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps Platform API...")
    settings = get_settings()
    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")
    logger.info(f"Default experiment: {settings.experiment_name}")
    logger.info(f"Default model: {settings.model_name}")
    yield
    logger.info("Shutting down MLOps Platform API...")
settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.middleware.security import limiter, SecurityMiddleware, get_api_key
from fastapi import Depends
Instrumentator().instrument(app).expose(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.include_router(experiments_router)
app.include_router(models_router)
app.include_router(train_router, dependencies=[Depends(get_api_key)])
app.include_router(predict_router, dependencies=[Depends(get_api_key)])
app.include_router(drift_router)
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc",
    }
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "mlops-api",
        "version": settings.api_version,
    }
@app.get("/ready", tags=["Health"])
async def readiness_check():
    try:
        from app.services.mlflow_service import get_mlflow_service
        mlflow_service = get_mlflow_service()
        mlflow_service.list_experiments()
        return {
            "status": "ready",
            "mlflow": "connected",
            "tracking_uri": settings.mlflow_tracking_uri,
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow connection failed: {str(e)}"
        )
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
