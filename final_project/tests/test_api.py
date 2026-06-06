import pytest

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


@pytest.mark.parametrize(
    ("text", "expected_label_id"),
    [
        ("我跟鄰居吵架，我被揍好痛好痛好痛", "injury"),
        ("我買了一台二手車，交車後才發現引擎有重大問題，賣家不願意退錢或修理。", "sales_defect"),
        ("我不小心把錢匯到陌生人的帳戶，對方知道後卻拒絕把錢還給我。", "unjust_enrichment"),
        ("前男友傳訊息說要讓我好看，叫我小心家人。", "intimidation"),
        ("我和廠商簽約後，對方收了訂金卻沒有依約完成工作。", "contract_breach"),
    ],
)
def test_predict_endpoint_handles_everyday_user_scenarios(text, expected_label_id):
    client = create_app().test_client()
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 200
    predicted_ids = {row["id"] for row in response.json["final_prediction"]}
    assert expected_label_id in predicted_ids
