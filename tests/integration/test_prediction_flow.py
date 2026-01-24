import pytest
from unittest.mock import patch, MagicMock


class TestPredictionFlowIntegration:

    def test_prediction_uses_production_model(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                "prediction": 7,
                "confidence": 0.98,
                "probabilities": [0.0] * 7 + [0.98] + [0.01, 0.01],
                "model_name": "MNISTClassifier",
                "model_version": "3",
                "model_stage": "Production",
            }
            mock_get_service.return_value = mock_service

            response = client.post(
                "/predict", json=sample_predict_request, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["model_stage"] == "Production"

    def test_prediction_returns_valid_class(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                "prediction": 5,
                "confidence": 0.92,
                "probabilities": [0.01] * 5 + [0.92] + [0.01] * 4,
                "model_name": "MNISTClassifier",
                "model_version": "1",
                "model_stage": "Production",
            }
            mock_get_service.return_value = mock_service

            response = client.post(
                "/predict", json=sample_predict_request, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert 0 <= data["prediction"] <= 9

    def test_prediction_includes_confidence(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                "prediction": 3,
                "confidence": 0.87,
                "probabilities": [0.02] * 3 + [0.87] + [0.02] * 6,
                "model_name": "MNISTClassifier",
                "model_version": "1",
                "model_stage": "Production",
            }
            mock_get_service.return_value = mock_service

            response = client.post(
                "/predict", json=sample_predict_request, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1

    def test_prediction_includes_probabilities(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                "prediction": 0,
                "confidence": 0.95,
                "probabilities": [0.95] + [0.005] * 9,
                "model_name": "MNISTClassifier",
                "model_version": "1",
                "model_stage": "Production",
            }
            mock_get_service.return_value = mock_service

            response = client.post(
                "/predict", json=sample_predict_request, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert "probabilities" in data
            assert len(data["probabilities"]) == 10

    def test_batch_prediction_works(self, client, sample_mnist_image, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict_batch.return_value = {
                "predictions": [1, 2, 3, 4, 5],
                "confidences": [0.9, 0.85, 0.92, 0.88, 0.95],
                "model_name": "MNISTClassifier",
                "model_version": "1",
                "batch_size": 5,
            }
            mock_get_service.return_value = mock_service

            response = client.post("/predict/batch", json={
                "images": [sample_mnist_image] * 5,
            }, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 5
            assert data["batch_size"] == 5


class TestPredictionErrorHandling:

    def test_no_production_model_error(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            with patch('app.routes.predict.redis.Redis') as mock_redis:
                mock_redis_instance = MagicMock()
                mock_redis_instance.get.return_value = None
                mock_redis.return_value = mock_redis_instance

                mock_service = MagicMock()
                mock_service.predict.side_effect = ValueError(
                    "No Production model found for 'MNISTClassifier'"
                )
                mock_get_service.return_value = mock_service

                response = client.post(
                    "/predict", json=sample_predict_request, headers=auth_headers)

                assert response.status_code == 404
                data = response.json()
                assert "detail" in data

    def test_staging_model_fallback(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                "prediction": 8,
                "confidence": 0.89,
                "probabilities": [0.01] * 8 + [0.89] + [0.02],
                "model_name": "MNISTClassifier",
                "model_version": "2",
                "model_stage": "Staging",
            }
            mock_get_service.return_value = mock_service

            response = client.post(
                "/predict/staging", json=sample_predict_request, headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["model_stage"] == "Staging"
