"""Build a statute-text lookup for the issue ontology.

Reads `lianghsun/tw-processed-law-article` and extracts the full article text for
every (law, article) pair referenced by `data/issues.yaml`. The result is written
to `data/law_articles.json` so the demo can show the actual statute content as a
lookup clue, not just the article number.

The dataset packs each row's `text` as "<law name> 第 <article> 條 <content>".
We parse the article key and keep the content after the heading.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from datasets import load_dataset

# Ontology short names -> full law names used in the statute dataset.
LAW_NAME_MAP = {
    "刑法": "中華民國刑法",
    "民法": "民法",
}

# Matches "第 185-3 條" / "第 320 條" and captures the article key.
ARTICLE_RE = re.compile(r"第\s*([0-9]+(?:-[0-9]+)?)\s*條")
# Splits the heading "<law> 第 <art> 條" from the article body.
HEADING_RE = re.compile(r"^.*?第\s*[0-9]+(?:-[0-9]+)?\s*條\s*")


def wanted_pairs(issues: list[dict]) -> set[tuple[str, str]]:
    """Collect (full_law_name, article) pairs we need from the ontology."""
    pairs: set[tuple[str, str]] = set()
    for issue in issues:
        for law in issue["laws"]:
            full = LAW_NAME_MAP.get(law["law"], law["law"])
            pairs.add((full, str(law["article"])))
    return pairs


def clean_body(text: str) -> str:
    """Drop the leading "<law> 第 N 條" heading, normalise whitespace."""
    body = HEADING_RE.sub("", text, count=1)
    return re.sub(r"\s+", " ", body).strip()


def main() -> None:
    issues = yaml.safe_load(Path("data/issues.yaml").read_text(encoding="utf-8"))
    pairs = wanted_pairs(issues)
    inverse = {v: k for k, v in LAW_NAME_MAP.items()}

    ds = load_dataset("lianghsun/tw-processed-law-article", split="train")
    found: dict[str, dict] = {}

    for row in ds:
        full_name = row["name"]
        if full_name not in inverse:
            continue
        match = ARTICLE_RE.search(row["text"])
        if not match:
            continue
        article = match.group(1)
        key_pair = (full_name, article)
        if key_pair not in pairs:
            continue
        short_law = inverse[full_name]
        key = f"{short_law}:{article}"
        if key in found:
            continue
        found[key] = {
            "law": short_law,
            "article": article,
            "full_law": full_name,
            "text": clean_body(row["text"]),
        }
        if len(found) == len(pairs):
            break

    missing = sorted(
        f"{inverse[name]}:{art}" for name, art in pairs if f"{inverse[name]}:{art}" not in found
    )

    output = Path("data/law_articles.json")
    output.write_text(json.dumps(found, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"resolved={len(found)}/{len(pairs)}")
    if missing:
        print("MISSING:", ", ".join(missing))


if __name__ == "__main__":
    main()
