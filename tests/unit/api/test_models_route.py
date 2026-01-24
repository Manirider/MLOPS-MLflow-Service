import pytest
from unittest.mock import patch, MagicMock

class TestModelsEndpoint:

    def test_list_models_success(self, client, mock_mlflow_service):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "total_count" in data

    def test_list_models_with_versions(self, client, mock_mlflow_service):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()

            if data["models"]:
                model = data["models"][0]
                assert "name" in model
                assert "latest_version" in model
                assert "versions" in model

    def test_list_models_empty(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_registered_models.return_value = []
            mock_get_service.return_value = mock_service

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["models"] == []
            assert data["total_count"] == 0

class TestModelDetailEndpoint:

    def test_get_model_success(self, client, mock_mlflow_service):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_get_service.return_value = mock_mlflow_service

            response = client.get("/models/MNISTClassifier")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "MNISTClassifier"

    def test_get_model_not_found(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_registered_models.return_value = []
            mock_get_service.return_value = mock_service

            response = client.get("/models/NonExistent")

            assert response.status_code == 404

class TestTransitionStageEndpoint:

    def test_transition_success(self, client):
        with patch('app.routes.models.get_mlflow_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            response = client.post("/models/transition", json={
                "model_name": "MNISTClassifier",
                "version": "1",
                "stage": "Production",
                "archive_existing": True,
            })

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "MNISTClassifier"
            assert data["stage"] == "Production"

    def test_transition_invalid_stage(self, client):
        response = client.post("/models/transition", json={
            "model_name": "MNISTClassifier",
            "version": "1",
            "stage": "InvalidStage",
        })

        assert response.status_code == 422
