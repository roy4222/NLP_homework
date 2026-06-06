"""Evaluate multi-label predictions and export Markdown result tables."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_predictions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_multilabel_metrics(
    gold: list[list[str]] | list[set[str]],
    predicted: list[list[str]] | list[set[str]],
    *,
    labels: list[str] | None = None,
) -> dict:
    if labels is None:
        labels = sorted({label for row in [*gold, *predicted] for label in row})
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(predicted)
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def compute_per_label_f1(
    gold: list[list[str]] | list[set[str]],
    predicted: list[list[str]] | list[set[str]],
    *,
    labels: list[str] | None = None,
) -> list[dict]:
    if labels is None:
        labels = sorted({label for row in [*gold, *predicted] for label in row})
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(gold)
    y_pred = mlb.transform(predicted)
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    supports = y_true.sum(axis=0)
    return [
        {"label": label, "f1": float(score), "support": int(support)}
        for label, score, support in zip(labels, scores, supports)
    ]


def score(name: str, rows: list[dict]) -> dict:
    gold = [row["gold"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    labels = sorted({label for row in [*gold, *predicted] for label in row})
    return {
        "method": name,
        **compute_multilabel_metrics(gold, predicted, labels=labels),
        "per_label_f1": compute_per_label_f1(gold, predicted, labels=labels),
    }


def write_report(results: list[dict], path: Path) -> None:
    lines = [
        "# Experimental Results",
        "",
        "| Method | Micro-F1 | Macro-F1 | Precision | Recall | Hamming Loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['method']} | {row['micro_f1']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['hamming_loss']:.3f} |"
        )
    lines.extend(["", "## Per-label F1", ""])
    for row in results:
        lines.extend(
            [
                f"### {row['method']}",
                "",
                "| Label | Support | F1 |",
                "|---|---:|---:|",
            ]
        )
        for item in sorted(row["per_label_f1"], key=lambda x: (x["f1"], x["support"], x["label"])):
            lines.append(f"| `{item['label']}` | {item['support']} | {item['f1']:.3f} |")
        lines.append("")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [
        score("Rule-based", load_predictions(Path("data/rule_predictions.json"))),
        score("TF-IDF + SVM", load_predictions(Path("data/tfidf_svm_predictions.json"))),
    ]
    write_report(results, Path("report/results.md"))


if __name__ == "__main__":
    main()
