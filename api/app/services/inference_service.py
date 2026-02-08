import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from app.config import get_settings
from app.services.mlflow_service import get_mlflow_service
logger = logging.getLogger(__name__)
class InferenceService:
    def __init__(self):
        self.settings = get_settings()
        self._model_cache: Dict[str, Any] = {}
        self._model_info_cache: Dict[str, Dict[str, Any]] = {}
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
    def _get_cache_key(self, model_name: str, stage: str) -> str:
        return f"{model_name}:{stage}"
    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
        force_reload: bool = False,
    ) -> Any:
        cache_key = self._get_cache_key(model_name, stage)
        if not force_reload and cache_key in self._model_cache:
            logger.debug(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]
        model_uri = f"models:/{model_name}/{stage}"
        try:
            logger.info(f"Loading model from: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            self._model_cache[cache_key] = model
            self._update_model_info_cache(model_name, stage)
            return model
        except MlflowException as e:
            logger.error(f"Failed to load model {model_uri}: {e}")
            raise ValueError(f"No {stage} model found for '{model_name}'")
    def _update_model_info_cache(self, model_name: str, stage: str) -> None:
        try:
            mlflow_service = get_mlflow_service()
            versions = mlflow_service.client.get_latest_versions(
                model_name, stages=[stage]
            )
            if versions:
                v = versions[0]
                cache_key = self._get_cache_key(model_name, stage)
                self._model_info_cache[cache_key] = {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
        except Exception as e:
            logger.warning(f"Failed to cache model info: {e}")
    def get_model_info(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(model_name, stage)
        return self._model_info_cache.get(cache_key)
    def predict(
        self,
        image_data: List[float],
        model_name: Optional[str] = None,
        stage: str = "Production",
    ) -> Dict[str, Any]:
        model_name = model_name or self.settings.model_name
        model = self.load_model(model_name, stage)
        input_array = np.array(image_data).reshape(1, -1)
        input_array = input_array / 255.0 if input_array.max() > 1.0 else input_array
        predictions = model.predict(input_array)
        model_info = self.get_model_info(model_name, stage) or {}
        if hasattr(predictions, 'tolist'):
            prediction = int(predictions[0])
            probabilities = [0.0] * 10
            probabilities[prediction] = 1.0
            confidence = 1.0
        else:
            prediction = int(predictions[0])
            probabilities = [0.0] * 10
            probabilities[prediction] = 1.0
            confidence = 1.0
        try:
            unwrapped = model._model_impl
            if hasattr(unwrapped, 'predict_proba'):
                proba = unwrapped.predict_proba(input_array)
                probabilities = proba[0].tolist()
                confidence = float(max(probabilities))
        except Exception:
            pass
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": model_name,
            "model_version": model_info.get("version", "unknown"),
            "model_stage": stage,
        }
    def predict_batch(
        self,
        images: List[List[float]],
        model_name: Optional[str] = None,
        stage: str = "Production",
    ) -> Dict[str, Any]:
        model_name = model_name or self.settings.model_name
        model = self.load_model(model_name, stage)
        input_array = np.array(images)
        input_array = input_array / 255.0 if input_array.max() > 1.0 else input_array
        predictions = model.predict(input_array)
        model_info = self.get_model_info(model_name, stage) or {}
        confidences = [1.0] * len(predictions)
        try:
            unwrapped = model._model_impl
            if hasattr(unwrapped, 'predict_proba'):
                proba = unwrapped.predict_proba(input_array)
                confidences = [float(max(p)) for p in proba]
        except Exception:
            pass
        return {
            "predictions": [int(p) for p in predictions],
            "confidences": confidences,
            "model_name": model_name,
            "model_version": model_info.get("version", "unknown"),
            "batch_size": len(images),
        }
    def clear_cache(self) -> None:
        self._model_cache.clear()
        self._model_info_cache.clear()
        logger.info("Model cache cleared")
_inference_service: Optional[InferenceService] = None
def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
