import pytest
from unittest.mock import patch, MagicMock

class TestExperimentsEndpoint:

    def test_list_experiments_success(self, client, mock_mlflow_service):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/experiments")

            assert response.status_code == 200
            data = response.json()
            assert "experiments" in data
            assert "total_count" in data
            assert data["total_count"] >= 1

    def test_list_experiments_with_best_run(self, client, mock_mlflow_service):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/experiments")

            assert response.status_code == 200
            data = response.json()

            if data["experiments"]:
                exp = data["experiments"][0]
                assert "experiment_id" in exp
                assert "name" in exp
                if exp.get("best_run"):
                    assert "run_id" in exp["best_run"]
                    assert "metrics" in exp["best_run"]

    def test_list_experiments_empty(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_experiments.return_value = []
            mock_get_service.return_value = mock_service

            response = client.get("/experiments")

            assert response.status_code == 200
            data = response.json()
            assert data["experiments"] == []
            assert data["total_count"] == 0

class TestExperimentDetailEndpoint:

    def test_get_experiment_success(self, client, mock_mlflow_service):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_mlflow_service.get_experiment_by_name.return_value = {
                "experiment_id": "1",
                "name": "MNIST_Experiments",
                "artifact_location": "/mlruns/1",
                "lifecycle_stage": "active",
            }
            mock_mlflow_service.search_runs.return_value = []
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/experiments/MNIST_Experiments")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "MNIST_Experiments"

    def test_get_experiment_not_found(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_experiment_by_name.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get("/experiments/NonExistent")

            assert response.status_code == 404

class TestRunDetailEndpoint:

    def test_get_run_success(self, client, mock_mlflow_service):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_mlflow_service.get_run.return_value = {
                "run_id": "abc123",
                "run_name": "test_run",
                "status": "FINISHED",
                "metrics": {"accuracy": 0.95},
                "params": {"learning_rate": "0.001"},
            }
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/experiments/MNIST/runs/abc123")

            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == "abc123"

    def test_get_run_not_found(self, client):
        with patch('app.routes.experiments.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_run.return_value = None
            mock_get_service.return_value = mock_service

            response = client.get("/experiments/MNIST/runs/invalid")

            assert response.status_code == 404
