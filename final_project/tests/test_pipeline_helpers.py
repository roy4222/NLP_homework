import json
from pathlib import Path

from scripts.build_labels import build_labeled_rows, load_issue_map
from scripts.split_data import split_rows


def test_build_labeled_rows_maps_citations_to_issue_ids(tmp_path: Path):
    ontology = tmp_path / "issues.yaml"
    ontology.write_text(
        """
- id: fraud
  laws:
    - law: 刑法
      article: "339"
""".strip(),
        encoding="utf-8",
    )
    rows = [
        {
            "messages": [
                {"role": "user", "content": "我在網路購物被對方騙走三萬元，對方收款後就封鎖我。"},
                {"role": "assistant", "content": "可能涉及刑法第339條詐欺罪。"},
            ]
        }
    ]

    labeled = build_labeled_rows(rows, load_issue_map(ontology))

    assert labeled == [
        {
            "id": "synthetic-0",
            "story": "我在網路購物被對方騙走三萬元，對方收款後就封鎖我。",
            "labels": ["fraud"],
            "citations": [["刑法", "339"]],
            "label_quality": "usable",
        }
    ]


def test_split_rows_is_deterministic_and_complete():
    rows = [{"id": str(i), "labels": ["x"]} for i in range(20)]

    first = split_rows(rows, seed=7)
    second = split_rows(rows, seed=7)

    assert first == second
    assert len(first["train"]) == 14
    assert len(first["val"]) == 3
    assert len(first["test"]) == 3
    assert sorted(int(row["id"]) for split in first.values() for row in split) == list(range(20))
