import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestMLflowTrackingIntegration:

    def test_experiment_created_on_training(self, client, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()

            class MockJob:
                job_id = "job_000001"
                run_id = "abc123"
                experiment_name = "Test_Experiments"
                status = "running"

            mock_service.start_training = AsyncMock(return_value=MockJob())
            mock_get_service.return_value = mock_service

            response = client.post("/train", json={
                "learning_rate": 0.001,
                "epochs": 2,
                "batch_size": 32,
            }, headers=auth_headers)

            assert response.status_code == 202

    def test_runs_searchable_after_training(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_experiments.return_value = [
                {
                    "experiment_id": "1",
                    "name": "MNIST_Experiments",
                    "total_runs": 5,
                    "best_run": {
                        "run_id": "run123",
                        "run_name": "test",
                        "status": "FINISHED",
                        "metrics": {"accuracy": 0.95},
                        "params": {},
                    }
                }
            ]
            mock_get_service.return_value = mock_service

            response = client.get("/experiments")

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] >= 1

    def test_metrics_logged_correctly(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_run.return_value = {
                "run_id": "abc123",
                "status": "FINISHED",
                "metrics": {
                    "accuracy": 0.95,
                    "precision_macro": 0.94,
                    "recall_macro": 0.93,
                    "f1_macro": 0.935,
                },
                "params": {
                    "learning_rate": "0.001",
                    "epochs": "10",
                },
            }
            mock_get_service.return_value = mock_service

            response = client.get("/experiments/MNIST/runs/abc123")

            assert response.status_code == 200
            data = response.json()
            assert "accuracy" in data["metrics"]
            assert data["metrics"]["accuracy"] > 0

    def test_parameters_logged_correctly(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_run.return_value = {
                "run_id": "abc123",
                "status": "FINISHED",
                "metrics": {},
                "params": {
                    "learning_rate": "0.001",
                    "epochs": "10",
                    "batch_size": "64",
                    "hidden_size": "128",
                },
            }
            mock_get_service.return_value = mock_service

            response = client.get("/experiments/MNIST/runs/abc123")

            assert response.status_code == 200
            data = response.json()
            assert "learning_rate" in data["params"]
            assert data["params"]["learning_rate"] == "0.001"


class TestMLflowConnectionHandling:

    def test_health_endpoint_available(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_with_mlflow_connection(self, client):
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
