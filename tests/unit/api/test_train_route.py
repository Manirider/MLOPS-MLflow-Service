import pytest
from unittest.mock import patch, MagicMock, AsyncMock
class TestTrainEndpoint:
    def test_train_valid_request(self, client, sample_train_request, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()
            class MockJob:
                job_id = "job_000001"
                run_id = None
                experiment_name = "Test_Experiments"
                status = "running"
            mock_service.start_training = AsyncMock(return_value=MockJob())
            mock_get_service.return_value = mock_service
            response = client.post(
                "/train", json=sample_train_request, headers=auth_headers)
            assert response.status_code == 202
            data = response.json()
            assert "message" in data
            assert data["status"] == "running"
    def test_train_default_values(self, client, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()
            class MockJob:
                job_id = "job_000001"
                run_id = None
                experiment_name = "MNIST_Experiments"
                status = "running"
            mock_service.start_training = AsyncMock(return_value=MockJob())
            mock_get_service.return_value = mock_service
            response = client.post("/train", json={}, headers=auth_headers)
            assert response.status_code == 202
    def test_train_invalid_learning_rate(self, client, auth_headers):
        response = client.post("/train", json={
            "learning_rate": 10.0,
        }, headers=auth_headers)
        assert response.status_code == 422
    def test_train_invalid_epochs(self, client, auth_headers):
        response = client.post("/train", json={
            "epochs": 0,
        }, headers=auth_headers)
        assert response.status_code == 422
    def test_train_invalid_batch_size(self, client, auth_headers):
        response = client.post("/train", json={
            "batch_size": 4,
        }, headers=auth_headers)
        assert response.status_code == 422
class TestTrainStatusEndpoint:
    def test_get_status_valid_job(self, client, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()
            class MockJob:
                job_id = "job_000001"
                run_id = "run_abc123"
                experiment_name = "MNIST_Experiments"
                status = "completed"
                started_at = None
                completed_at = None
                error = None
                params = {"learning_rate": 0.001}
            mock_service.get_job.return_value = MockJob()
            mock_get_service.return_value = mock_service
            response = client.get(
                "/train/status/job_000001", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "job_000001"
    def test_get_status_not_found(self, client, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_job.return_value = None
            mock_get_service.return_value = mock_service
            response = client.get(
                "/train/status/invalid_job", headers=auth_headers)
            assert response.status_code == 404
class TestListJobsEndpoint:
    def test_list_jobs(self, client, auth_headers):
        with patch('app.routes.train.get_training_service') as mock_get_service:
            mock_service = MagicMock()
            class MockJob:
                job_id = "job_000001"
                run_id = "run_abc123"
                experiment_name = "MNIST_Experiments"
                status = "completed"
                started_at = None
            mock_service.list_jobs.return_value = [MockJob()]
            mock_get_service.return_value = mock_service
            response = client.get("/train/jobs", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert "jobs" in data
            assert data["total_count"] == 1
