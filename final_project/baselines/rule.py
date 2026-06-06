"""Rule-based baseline using ontology aliases and keywords."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_one(story: str, issues: list[dict]) -> list[dict]:
    predictions = []
    for issue in issues:
        terms = list(issue.get("aliases", [])) + list(issue.get("keywords", []))
        hits = sorted({term for term in terms if term and term in story})
        if hits:
            predictions.append(
                {
                    "id": issue["id"],
                    "name_zh": issue["name_zh"],
                    "name_en": issue["name_en"],
                    "confidence": min(0.95, 0.45 + 0.1 * len(hits)),
                    "supporting_laws": issue["laws"],
                    "evidence": hits[:5],
                }
            )
    return sorted(predictions, key=lambda item: item["confidence"], reverse=True)


def main() -> None:
    issues = yaml.safe_load(Path("data/issues.yaml").read_text(encoding="utf-8"))
    rows = read_jsonl(Path("data/test.jsonl"))
    output = [
        {
            "id": row["id"],
            "gold": row["labels"],
            "predicted": [item["id"] for item in predict_one(row["story"], issues)],
            "details": predict_one(row["story"], issues),
        }
        for row in rows
    ]
    Path("data/rule_predictions.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
