def test_predict_invalid_payload(client, auth_headers):
    response = client.post(
        "/predict", json={"wrong": []}, headers=auth_headers)
    assert response.status_code == 422
