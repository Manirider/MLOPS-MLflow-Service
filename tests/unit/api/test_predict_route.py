import pytest
from unittest.mock import patch, MagicMock
class TestPredictEndpoint:
    def test_predict_success(self, client, sample_predict_request, mock_inference_service, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_get_service.return_value = mock_inference_service
            response = client.post(
                "/predict", json=sample_predict_request, headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "model_name" in data
            assert "model_version" in data
            assert 0 <= data["prediction"] <= 9
    def test_predict_invalid_image_size(self, client, auth_headers):
        response = client.post("/predict", json={
            "image": [0.0] * 100,
        }, headers=auth_headers)
        assert response.status_code == 422
    def test_predict_invalid_pixel_values(self, client, auth_headers):
        response = client.post("/predict", json={
            "image": [-1.0] + [0.0] * 783,
        }, headers=auth_headers)
        assert response.status_code == 422
    def test_predict_no_production_model(self, client, sample_predict_request, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            with patch('app.routes.predict.redis.Redis') as mock_redis:
                mock_redis_instance = MagicMock()
                mock_redis_instance.get.return_value = None
                mock_redis.return_value = mock_redis_instance
                mock_service = MagicMock()
                mock_service.predict.side_effect = ValueError(
                    "No Production model found")
                mock_get_service.return_value = mock_service
                response = client.post(
                    "/predict", json=sample_predict_request, headers=auth_headers)
                assert response.status_code == 404
class TestBatchPredictEndpoint:
    def test_batch_predict_success(self, client, sample_mnist_image, mock_inference_service, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_get_service.return_value = mock_inference_service
            response = client.post("/predict/batch", json={
                "images": [sample_mnist_image, sample_mnist_image, sample_mnist_image],
            }, headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "confidences" in data
            assert "batch_size" in data
    def test_batch_predict_empty(self, client, auth_headers):
        response = client.post("/predict/batch", json={
            "images": [],
        }, headers=auth_headers)
        assert response.status_code == 422
class TestStagingPredictEndpoint:
    def test_predict_staging_success(self, client, sample_predict_request, mock_inference_service, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_get_service.return_value = mock_inference_service
            response = client.post(
                "/predict/staging", json=sample_predict_request, headers=auth_headers)
            assert response.status_code == 200
class TestClearCacheEndpoint:
    def test_clear_cache_success(self, client, auth_headers):
        with patch('app.routes.predict.get_inference_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            response = client.delete("/predict/cache", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
