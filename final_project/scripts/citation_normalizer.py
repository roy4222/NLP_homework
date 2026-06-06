"""Normalize Taiwan legal citations into stable (law_name, article_key) pairs."""

from __future__ import annotations

import re
import unicodedata

LAW_ALIASES = {
    "中華民國刑法": "刑法",
    "刑事法": "刑法",
    "中華民國民法": "民法",
    "臺灣民法": "民法",
    "台灣民法": "民法",
}

LAW_NAMES = sorted(
    {
        "中華民國刑法",
        "中華民國民法",
        "臺灣民法",
        "台灣民法",
        "刑法",
        "民法",
        "刑事訴訟法",
        "民事訴訟法",
        "道路交通管理處罰條例",
        "個人資料保護法",
    },
    key=len,
    reverse=True,
)

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

ARTICLE_NUMBER = r"[0-9零〇一二兩三四五六七八九十百千]+"
CITATION_RE = re.compile(
    r"(" + "|".join(re.escape(name) for name in LAW_NAMES) + r")"
    r"\s*第?\s*(" + ARTICLE_NUMBER + r")\s*條"
    r"(?:\s*(?:之|-)\s*(" + ARTICLE_NUMBER + r"))?"
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized)


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


def normalize_law_name(law_name: str) -> str:
    return LAW_ALIASES.get(law_name.strip(), law_name.strip())


def article_key(main: str, sub: str | None) -> str:
    key = str(cn_to_int(main))
    if sub:
        return f"{key}-{cn_to_int(sub)}"
    return key


def extract_citations(text: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for law_name, main, sub in CITATION_RE.findall(normalize_text(text)):
        pairs.add((normalize_law_name(law_name), article_key(main, sub or None)))
    return pairs
