import pytest
from unittest.mock import patch, MagicMock

class TestModelRegistryIntegration:

    def test_model_registered_after_training(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_registered_models.return_value = [
                {
                    "name": "MNISTClassifier",
                    "description": "MNIST digit classifier",
                    "latest_version": "1",
                    "latest_stage": "Staging",
                    "versions": [
                        {
                            "version": "1",
                            "stage": "Staging",
                            "run_id": "abc123",
                            "status": "READY",
                        }
                    ],
                }
            ]
            mock_get_service.return_value = mock_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] >= 1

            model = data["models"][0]
            assert model["name"] == "MNISTClassifier"

    def test_model_version_increments(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_registered_models.return_value = [
                {
                    "name": "MNISTClassifier",
                    "latest_version": "3",
                    "latest_stage": "Production",
                    "versions": [
                        {"version": "1", "stage": "Archived", "run_id": "run1", "status": "READY"},
                        {"version": "2", "stage": "Archived", "run_id": "run2", "status": "READY"},
                        {"version": "3", "stage": "Production", "run_id": "run3", "status": "READY"},
                    ],
                }
            ]
            mock_get_service.return_value = mock_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["models"][0]["latest_version"] == "3"

    def test_stage_transition_to_staging(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            response = client.post("/models/transition", json={
                "model_name": "MNISTClassifier",
                "version": "1",
                "stage": "Staging",
            })

            assert response.status_code == 200
            data = response.json()
            assert data["stage"] == "Staging"

    def test_stage_transition_to_production(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            response = client.post("/models/transition", json={
                "model_name": "MNISTClassifier",
                "version": "1",
                "stage": "Production",
            })

            assert response.status_code == 200
            data = response.json()
            assert data["stage"] == "Production"

    def test_multiple_models_supported(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_registered_models.return_value = [
                {"name": "Model1", "latest_version": "1", "latest_stage": "Production", "versions": []},
                {"name": "Model2", "latest_version": "2", "latest_stage": "Staging", "versions": []},
                {"name": "Model3", "latest_version": "1", "latest_stage": "None", "versions": []},
            ]
            mock_get_service.return_value = mock_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 3
