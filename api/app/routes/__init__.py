from app.routes.train import router as train_router
from app.routes.experiments import router as experiments_router
from app.routes.models import router as models_router
from app.routes.predict import router as predict_router
from app.routes.drift import router as drift_router
__all__ = [
    "train_router",
    "experiments_router",
    "models_router",
    "predict_router",
    "drift_router",
]
