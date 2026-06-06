from baselines.rule import predict_one
from baselines.tfidf_svm import train_and_predict


ISSUES = [
    {
        "id": "fraud",
        "name_zh": "詐欺",
        "name_en": "fraud",
        "laws": [{"law": "刑法", "article": "339"}],
        "aliases": ["詐欺"],
        "keywords": ["騙", "詐騙"],
    },
    {
        "id": "theft",
        "name_zh": "竊盜",
        "name_en": "theft",
        "laws": [{"law": "刑法", "article": "320"}],
        "aliases": ["竊盜"],
        "keywords": ["偷", "竊取"],
    },
]


def test_rule_baseline_returns_keyword_evidence():
    predictions = predict_one("對方在網路詐騙我，騙走匯款。", ISSUES)
    assert predictions[0]["id"] == "fraud"
    assert "騙" in predictions[0]["evidence"]


def test_tfidf_svm_trains_and_predicts_multilabel_rows():
    train = [
        {"id": "1", "story": "網路詐騙匯款被騙", "labels": ["fraud"]},
        {"id": "2", "story": "手機被偷遭人竊取", "labels": ["theft"]},
        {"id": "3", "story": "投資詐騙對方失聯", "labels": ["fraud"]},
        {"id": "4", "story": "腳踏車被偷走", "labels": ["theft"]},
    ]
    test = [{"id": "t", "story": "我被詐騙匯款", "labels": ["fraud"]}]

    predictions, _bundle = train_and_predict(train, test, min_df=1)

    assert predictions[0]["gold"] == ["fraud"]
    assert "fraud" in predictions[0]["predicted"]
