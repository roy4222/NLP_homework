"""Prediction service for the Flask API."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import yaml

from baselines.rule import predict_one

DISCLAIMER = "This tool provides legal issue triage for NLP demonstration only. It is not legal advice."

# How many candidate issues to surface in the per-issue comparison grid.
GRID_SIZE = 8
# Decision-function sigmoid floor for an issue to enter the grid as a TF-IDF candidate.
GRID_FLOOR = 0.30


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class ModelService:
    def __init__(
        self,
        ontology_path: Path = Path("data/issues.yaml"),
        model_path: Path = Path("data/tfidf_svm.joblib"),
        law_path: Path = Path("data/law_articles.json"),
    ) -> None:
        self.issues = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
        self.by_id = {issue["id"]: issue for issue in self.issues}
        self.model_bundle = joblib.load(model_path) if model_path.exists() else None
        self.law_lookup: dict[str, dict] = (
            json.loads(law_path.read_text(encoding="utf-8")) if law_path.exists() else {}
        )

    # ---- statute text enrichment -------------------------------------------
    def _enrich_laws(self, laws: list[dict]) -> list[dict]:
        """Attach the full statute text to each {law, article} reference."""
        enriched = []
        for law in laws:
            key = f"{law['law']}:{law['article']}"
            entry = self.law_lookup.get(key)
            enriched.append(
                {
                    "law": law["law"],
                    "article": str(law["article"]),
                    "text": entry["text"] if entry else "",
                    "full_law": entry["full_law"] if entry else law["law"],
                }
            )
        return enriched

    def _laws_for(self, issue_id: str) -> list[dict]:
        issue = self.by_id.get(issue_id)
        return self._enrich_laws(issue["laws"]) if issue else []

    # ---- per-model scoring --------------------------------------------------
    def _tfidf_scores(self, text: str) -> dict[str, float]:
        """Sigmoid-mapped decision-function score per issue id (0..1)."""
        if not self.model_bundle:
            return {}
        pipeline = self.model_bundle["pipeline"]
        mlb = self.model_bundle["mlb"]
        scores = pipeline.decision_function([text])[0]
        return {label: _sigmoid(float(score)) for label, score in zip(mlb.classes_, scores)}

    def _tfidf_predictions(self, scores: dict[str, float]) -> list[dict]:
        rows = []
        for issue_id, score in scores.items():
            if score < 0.5:
                continue
            issue = self.by_id.get(issue_id)
            if not issue:
                continue
            rows.append(
                {
                    "id": issue["id"],
                    "name_zh": issue["name_zh"],
                    "name_en": issue["name_en"],
                    "confidence": round(score, 2),
                    "supporting_laws": self._laws_for(issue_id),
                    "evidence": ["tfidf_svm"],
                }
            )
        return sorted(rows, key=lambda item: item["confidence"], reverse=True)

    def _build_grid(
        self, rule_predictions: list[dict], tfidf_scores: dict[str, float]
    ) -> list[dict]:
        """Candidate issues with per-model scores for the comparison chart."""
        rule_ids = {row["id"] for row in rule_predictions}
        candidates = set(rule_ids)
        candidates |= {i for i, s in tfidf_scores.items() if s >= GRID_FLOOR}
        grid = []
        for issue_id in candidates:
            issue = self.by_id.get(issue_id)
            if not issue:
                continue
            grid.append(
                {
                    "id": issue_id,
                    "name_zh": issue["name_zh"],
                    "name_en": issue["name_en"],
                    "rule": 1 if issue_id in rule_ids else 0,
                    "tfidf_svm": round(tfidf_scores.get(issue_id, 0.0), 2),
                }
            )
        # Strongest first: TF-IDF score, then rule hit.
        grid.sort(key=lambda row: (row["tfidf_svm"], row["rule"]), reverse=True)
        return grid[:GRID_SIZE]

    # ---- main entry point ---------------------------------------------------
    def predict(self, text: str) -> dict:
        rule_predictions = predict_one(text, self.issues)
        for prediction in rule_predictions:
            prediction["supporting_laws"] = self._laws_for(prediction["id"])

        tfidf_scores = self._tfidf_scores(text)
        tfidf_predictions = self._tfidf_predictions(tfidf_scores)

        merged = {prediction["id"]: prediction for prediction in tfidf_predictions}
        for prediction in rule_predictions:
            merged[prediction["id"]] = prediction
        final_predictions = sorted(
            merged.values(), key=lambda item: item["confidence"], reverse=True
        )[:5]
        needs_review = not final_predictions or final_predictions[0]["confidence"] < 0.6

        return {
            "input": text,
            "status": "ok" if final_predictions else "no_match",
            "final_prediction": final_predictions,
            "models": {
                "rule": rule_predictions[:5],
                "tfidf_svm": tfidf_predictions[:5],
                "bert": [],  # not trained; see report (optional future work)
            },
            "grid": self._build_grid(rule_predictions, tfidf_scores),
            "explanation": {
                "matched_keywords": sorted(
                    {hit for pred in rule_predictions for hit in pred["evidence"]}
                ),
                "needs_review": needs_review,
            },
            "disclaimer": DISCLAIMER,
        }
