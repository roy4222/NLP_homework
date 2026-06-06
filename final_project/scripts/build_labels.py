"""Build weak labels from tw-legal-synthetic-qa using the issue ontology."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.citation_normalizer import extract_citations


def load_issue_map(path: Path) -> dict[tuple[str, str], list[str]]:
    issues = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], list[str]] = {}
    for issue in issues:
        for law in issue["laws"]:
            key = (str(law["law"]), str(law["article"]))
            mapping.setdefault(key, []).append(issue["id"])
    return mapping


def get_message(row: dict, role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def build_labeled_rows(
    rows: Iterable[dict],
    issue_map: dict[tuple[str, str], list[str]],
    *,
    min_story_length: int = 10,
) -> list[dict]:
    labeled = []
    for idx, row in enumerate(rows):
        story = get_message(row, "user").strip()
        analysis = get_message(row, "assistant").strip()
        citations = sorted(extract_citations(analysis))
        labels = sorted({label for pair in citations for label in issue_map.get(pair, [])})
        if not story or len(story) < min_story_length or not labels:
            continue
        labeled.append(
            {
                "id": f"synthetic-{idx}",
                "story": story,
                "labels": labels,
                "citations": [[law, article] for law, article in citations],
                "label_quality": "usable",
            }
        )
    return labeled


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    from datasets import load_dataset

    issue_map = load_issue_map(Path("data/issues.yaml"))
    ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
    rows = list(ds["train"]) + list(ds["test"])
    labeled = build_labeled_rows(rows, issue_map, min_story_length=20)
    write_jsonl(Path("data/labeled.jsonl"), labeled)

    label_counter: Counter[str] = Counter(label for row in labeled for label in row["labels"])
    print(f"kept={len(labeled)}")
    print(f"discarded={len(rows) - len(labeled)}")
    for label, count in label_counter.most_common():
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
