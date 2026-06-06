from api.app import create_app


def test_health_endpoint():
    client = create_app().test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_predict_endpoint_returns_labels_and_disclaimer():
    client = create_app().test_client()
    response = client.post("/api/predict", json={"text": "我在網路上被詐騙，匯款後對方封鎖我。"})
    assert response.status_code == 200
    assert response.json["input"]
    assert response.json["final_prediction"]
    assert response.json["disclaimer"]
    assert "not legal advice" in response.json["disclaimer"].lower()


def test_predict_endpoint_rejects_short_input():
    client = create_app().test_client()
    response = client.post("/api/predict", json={"text": "詐騙"})
    assert response.status_code == 400
    assert response.json["status"] == "error"
