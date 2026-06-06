"""Prediction service for the Flask API."""

from __future__ import annotations

from pathlib import Path

import joblib
import yaml

from baselines.rule import predict_one

DISCLAIMER = "This tool provides legal issue triage for NLP demonstration only. It is not legal advice."


class ModelService:
    def __init__(
        self,
        ontology_path: Path = Path("data/issues.yaml"),
        model_path: Path = Path("data/tfidf_svm.joblib"),
    ) -> None:
        self.issues = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
        self.model_bundle = joblib.load(model_path) if model_path.exists() else None

    def _predict_tfidf(self, text: str) -> list[dict]:
        if not self.model_bundle:
            return []
        pipeline = self.model_bundle["pipeline"]
        mlb = self.model_bundle["mlb"]
        predicted = mlb.inverse_transform(pipeline.predict([text]))[0]
        by_id = {issue["id"]: issue for issue in self.issues}
        rows = []
        for issue_id in predicted:
            issue = by_id.get(issue_id)
            if not issue:
                continue
            rows.append(
                {
                    "id": issue["id"],
                    "name_zh": issue["name_zh"],
                    "name_en": issue["name_en"],
                    "confidence": 0.65,
                    "supporting_laws": issue["laws"],
                    "evidence": ["tfidf_svm"],
                }
            )
        return rows

    def predict(self, text: str) -> dict:
        rule_predictions = predict_one(text, self.issues)
        tfidf_predictions = self._predict_tfidf(text)
        merged = {prediction["id"]: prediction for prediction in tfidf_predictions}
        for prediction in rule_predictions:
            merged[prediction["id"]] = prediction
        final_predictions = sorted(merged.values(), key=lambda item: item["confidence"], reverse=True)[:5]
        needs_review = not final_predictions or final_predictions[0]["confidence"] < 0.6
        return {
            "input": text,
            "status": "ok",
            "final_prediction": final_predictions,
            "models": {
                "rule": rule_predictions[:5],
                "tfidf_svm": tfidf_predictions[:5],
                "bert": [],
            },
            "explanation": {
                "matched_keywords": sorted({hit for pred in rule_predictions for hit in pred["evidence"]}),
                "needs_review": needs_review,
            },
            "disclaimer": DISCLAIMER,
        }
