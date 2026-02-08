import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient
@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    os.environ.setdefault("EXPERIMENT_NAME", "Test_Experiments")
    os.environ.setdefault("MODEL_NAME", "TestModel")
    os.environ.setdefault("API_KEY", "mlops-secret-key-123")
    yield
@pytest.fixture
def app():
    from app.main import app as fastapi_app
    return fastapi_app
@pytest.fixture
def client(app):
    return TestClient(app)
@pytest.fixture
def auth_headers():
    return {"X-API-Key": "mlops-secret-key-123"}
@pytest.fixture
def mock_mlflow_service():
    mock = MagicMock()
    mock.list_experiments.return_value = [
        {
            "experiment_id": "1",
            "name": "MNIST_Experiments",
            "artifact_location": "/mlruns/1",
            "lifecycle_stage": "active",
            "total_runs": 5,
            "best_run": {
                "run_id": "abc123",
                "run_name": "best_run",
                "status": "FINISHED",
                "metrics": {"accuracy": 0.95, "f1_macro": 0.94},
                "params": {"learning_rate": "0.001", "epochs": "10"},
            },
        }
    ]
    mock.list_registered_models.return_value = [
        {
            "name": "MNISTClassifier",
            "description": "Test model",
            "latest_version": "1",
            "latest_stage": "Production",
            "versions": [
                {
                    "version": "1",
                    "stage": "Production",
                    "run_id": "abc123",
                    "status": "READY",
                }
            ],
        }
    ]
    mock.get_production_model.return_value = {
        "name": "MNISTClassifier",
        "version": "1",
        "stage": "Production",
        "run_id": "abc123",
    }
    return mock
@pytest.fixture
def mock_training_service():
    mock = MagicMock()
    class MockJob:
        job_id = "job_000001"
        run_id = "run_abc123"
        experiment_name = "MNIST_Experiments"
        status = "running"
        started_at = None
        completed_at = None
        error = None
        params = {"learning_rate": 0.001}
    mock.start_training = AsyncMock(return_value=MockJob())
    mock.get_job.return_value = MockJob()
    mock.list_jobs.return_value = [MockJob()]
    return mock
@pytest.fixture
def mock_inference_service():
    mock = MagicMock()
    mock.predict.return_value = {
        "prediction": 7,
        "confidence": 0.98,
        "probabilities": [0.01] * 7 + [0.98] + [0.005, 0.005],
        "model_name": "MNISTClassifier",
        "model_version": "1",
        "model_stage": "Production",
    }
    mock.predict_batch.return_value = {
        "predictions": [7, 3, 5],
        "confidences": [0.98, 0.95, 0.92],
        "model_name": "MNISTClassifier",
        "model_version": "1",
        "batch_size": 3,
    }
    return mock
@pytest.fixture
def sample_mnist_image():
    import numpy as np
    np.random.seed(42)
    return (np.random.rand(784) * 255).tolist()
@pytest.fixture
def sample_train_request():
    return {
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 64,
        "hidden_size": 128,
        "dropout": 0.2,
        "experiment_name": "Test_Experiments",
        "run_name": "test_run",
    }
@pytest.fixture
def sample_predict_request(sample_mnist_image):
    return {"image": sample_mnist_image}
