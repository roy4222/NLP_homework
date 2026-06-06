"""TF-IDF + One-vs-Rest Linear SVM for multi-label issue classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_and_predict(
    train: list[dict],
    test: list[dict],
    *,
    min_df: int = 2,
) -> tuple[list[dict], dict]:
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([row["labels"] for row in train])
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=min_df)),
            ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced", dual="auto"))),
        ]
    )
    pipeline.fit([row["story"] for row in train], y_train)
    y_pred = pipeline.predict([row["story"] for row in test])
    labels = mlb.inverse_transform(y_pred)
    predictions = [
        {"id": row["id"], "gold": row["labels"], "predicted": sorted(list(pred))}
        for row, pred in zip(test, labels)
    ]
    return predictions, {"pipeline": pipeline, "mlb": mlb}


def main() -> None:
    predictions, bundle = train_and_predict(read_jsonl(Path("data/train.jsonl")), read_jsonl(Path("data/test.jsonl")))
    joblib.dump(bundle, "data/tfidf_svm.joblib")
    Path("data/tfidf_svm_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
