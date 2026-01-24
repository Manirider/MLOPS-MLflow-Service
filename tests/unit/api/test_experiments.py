def test_experiments_endpoint(client):
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert "total_count" in data
