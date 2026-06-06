"""Count high-frequency cited articles in tw-legal-synthetic-qa.

This script focuses on citations in the assistant's legal analysis, because
those citations are the weak-label source for the final project.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from datasets import load_dataset


TARGET_LAWS = {"刑法", "民法", "刑事訴訟法"}

LAW_ALIASES = {
    "中華民國刑法": "刑法",
    "中華民國民法": "民法",
    "中華民國刑事訴訟法": "刑事訴訟法",
}

CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

CITATION_RE = re.compile(
    r"(中華民國刑法|中華民國民法|中華民國刑事訴訟法|刑事訴訟法|刑法|民法)"
    r"\s*第\s*([0-9零一二三四五六七八九十百千]+)"
    r"\s*條(?:\s*之\s*([0-9零一二三四五六七八九十百千]+))?"
)


def cn_to_int(text: str) -> int:
    text = text.strip()
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    for char in text:
        if char in CN_DIGITS:
            current = CN_DIGITS[char]
        elif char == "十":
            total += (current or 1) * 10
            current = 0
        elif char == "百":
            total += (current or 1) * 100
            current = 0
        elif char == "千":
            total += (current or 1) * 1000
            current = 0
    return total + current


def assistant_text(row: dict) -> str:
    for message in row["messages"]:
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("臺", "台")


def article_key(main: str, sub: str | None) -> str:
    base = str(cn_to_int(main))
    if sub:
        return f"{base}-{cn_to_int(sub)}"
    return base


def main() -> None:
    ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
    rows = list(ds["train"]) + list(ds["test"])

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    rows_with_law = Counter()

    for row in rows:
        text = normalize_text(assistant_text(row))
        row_pairs = set()
        for law, main_no, sub_no in CITATION_RE.findall(text):
            law = LAW_ALIASES.get(law, law)
            if law not in TARGET_LAWS:
                continue
            key = article_key(main_no, sub_no or None)
            row_pairs.add((law, key))
        for law, key in row_pairs:
            counters[law][key] += 1
        for law in {law for law, _ in row_pairs}:
            rows_with_law[law] += 1

    print(f"total_rows\t{len(rows)}")
    for law in ["刑法", "民法", "刑事訴訟法"]:
        print(f"\n## {law}")
        print(f"rows_with_law\t{rows_with_law[law]}")
        print(f"unique_articles\t{len(counters[law])}")
        for article, count in counters[law].most_common(30):
            print(f"{law}第{article.replace('-', '條之')}條\t{count}")


if __name__ == "__main__":
    main()
