import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter
from app.models.prediction_log import PredictionLog, get_session
logger = logging.getLogger(__name__)
class DriftService:
    PSI_NO_DRIFT = 0.1
    PSI_MODERATE_DRIFT = 0.2
    PSI_SIGNIFICANT_DRIFT = 0.25
    def __init__(self):
        self._baseline_distribution: Optional[Dict[int, float]] = None
        self._baseline_count: int = 0
    def log_prediction(
        self,
        model_name: str,
        input_features: List[float],
        prediction: int,
        confidence: float,
        model_version: Optional[str] = None,
        request_id: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        try:
            session = get_session()
            log_entry = PredictionLog(
                model_name=model_name,
                model_version=model_version,
                input_features=input_features,
                prediction=prediction,
                confidence=confidence,
                request_id=request_id,
                latency_ms=latency_ms
            )
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    def get_prediction_distribution(
        self,
        model_name: str,
        hours: int = 24
    ) -> Dict[int, float]:
        try:
            session = get_session()
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            predictions = session.query(PredictionLog.prediction).filter(
                PredictionLog.model_name == model_name,
                PredictionLog.timestamp >= cutoff
            ).all()
            session.close()
            if not predictions:
                return {}
            pred_list = [p[0] for p in predictions]
            counts = Counter(pred_list)
            total = len(pred_list)
            distribution = {k: v / total for k, v in counts.items()}
            return distribution
        except Exception as e:
            logger.error(f"Failed to get distribution: {e}")
            return {}
    def set_baseline(self, distribution: Dict[int, float], count: int) -> None:
        self._baseline_distribution = distribution
        self._baseline_count = count
    def calculate_psi(
        self,
        baseline: Dict[int, float],
        current: Dict[int, float],
        num_classes: int = 10
    ) -> float:
        epsilon = 1e-10
        psi = 0.0
        for i in range(num_classes):
            baseline_pct = baseline.get(i, epsilon)
            current_pct = current.get(i, epsilon)
            baseline_pct = max(baseline_pct, epsilon)
            current_pct = max(current_pct, epsilon)
            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        return abs(psi)
    def get_drift_status(
        self,
        model_name: str,
        baseline_hours: int = 168,
        current_hours: int = 24
    ) -> Dict:
        baseline_dist = self.get_prediction_distribution(model_name, hours=baseline_hours)
        current_dist = self.get_prediction_distribution(model_name, hours=current_hours)
        if not baseline_dist or not current_dist:
            return {
                "status": "insufficient_data",
                "psi": None,
                "message": "Not enough prediction data for drift analysis",
                "baseline_samples": 0,
                "current_samples": 0
            }
        psi = self.calculate_psi(baseline_dist, current_dist)
        if psi < self.PSI_NO_DRIFT:
            status = "stable"
            message = "No significant drift detected"
        elif psi < self.PSI_MODERATE_DRIFT:
            status = "warning"
            message = "Moderate drift detected - monitor closely"
        else:
            status = "alert"
            message = "Significant drift detected - consider retraining"
        return {
            "status": status,
            "psi": round(psi, 4),
            "message": message,
            "baseline_distribution": baseline_dist,
            "current_distribution": current_dist,
            "thresholds": {
                "no_drift": self.PSI_NO_DRIFT,
                "moderate": self.PSI_MODERATE_DRIFT,
                "significant": self.PSI_SIGNIFICANT_DRIFT
            }
        }
    def get_prediction_stats(
        self,
        model_name: str,
        hours: int = 24
    ) -> Dict:
        try:
            session = get_session()
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            logs = session.query(PredictionLog).filter(
                PredictionLog.model_name == model_name,
                PredictionLog.timestamp >= cutoff
            ).all()
            session.close()
            if not logs:
                return {"total_predictions": 0}
            confidences = [log.confidence for log in logs]
            latencies = [log.latency_ms for log in logs if log.latency_ms]
            return {
                "total_predictions": len(logs),
                "avg_confidence": round(np.mean(confidences), 4),
                "min_confidence": round(min(confidences), 4),
                "max_confidence": round(max(confidences), 4),
                "avg_latency_ms": round(np.mean(latencies), 2) if latencies else None,
                "time_window_hours": hours
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
_drift_service: Optional[DriftService] = None
def get_drift_service() -> DriftService:
    global _drift_service
    if _drift_service is None:
        _drift_service = DriftService()
    return _drift_service
