"""Split labeled data into train/val/test and write dataset statistics."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def split_rows(rows: list[dict], *, seed: int = 42) -> dict[str, list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def label_counts(rows: list[dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row["labels"])
    return counter


def write_stats(path: Path, splits: dict[str, list[dict]]) -> None:
    rows = [row for split in splits.values() for row in split]
    lines = [
        "# Dataset Statistics",
        "",
        f"- Total labeled rows: {len(rows)}",
        f"- Train: {len(splits['train'])}",
        f"- Validation: {len(splits['val'])}",
        f"- Test: {len(splits['test'])}",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in label_counts(rows).most_common():
        lines.append(f"| `{label}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    splits = split_rows(read_jsonl(Path("data/labeled.jsonl")))
    for name, rows in splits.items():
        write_jsonl(Path(f"data/{name}.jsonl"), rows)
    write_stats(Path("data/stats.md"), splits)


if __name__ == "__main__":
    main()
