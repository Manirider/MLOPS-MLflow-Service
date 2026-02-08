def test_train_endpoint(client, auth_headers):
    response = client.post(
        "/train",
        json={"learning_rate": 0.001, "epochs": 1},
        headers=auth_headers
    )
    assert response.status_code == 202
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"
