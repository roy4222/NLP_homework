"""Count law names in tw-processed-law-article.

This verifies whether a law has many rows in the statute corpus. It does not
measure how often that law appears in legal scenarios.
"""

from __future__ import annotations

from collections import Counter
import re

from datasets import load_dataset


def show(title: str, counter: Counter[str], limit: int = 20) -> None:
    print(f"\n## {title}")
    for i, (name, count) in enumerate(counter.most_common(limit), start=1):
        print(f"{i:02d}\t{name}\t{count}")


def main() -> None:
    ds = load_dataset("lianghsun/tw-processed-law-article", split="train")
    print(f"total_rows\t{len(ds)}")
    print(f"columns\t{ds.column_names}")

    all_names: Counter[str] = Counter()
    active_names: Counter[str] = Counter()
    article_level_names: Counter[str] = Counter()
    active_article_level_names: Counter[str] = Counter()
    target_names: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    rough_topics = {
        "交通/運輸/航港": re.compile("交通|道路|車|船|航|港|鐵路|運輸|飛航|航空"),
        "金融/銀行/保險/證券": re.compile("金融|銀行|保險|證券|期貨|信用|票券|信託"),
        "醫療/衛生/食品藥物": re.compile("醫|衛生|食品|藥|傳染病|健康|護理|醫療"),
        "教育/學校": re.compile("教育|學校|學生|教師|大學|高級中等|國民教育"),
        "司法/訴訟/刑事": re.compile("司法|法院|訴訟|刑事|民事|行政訴訟|檢察|監獄|矯正"),
        "勞動/勞工/職安": re.compile("勞動|勞工|職業安全|職安|就業|工會|工資|職災"),
    }
    rough_topic_counts: Counter[str] = Counter()

    for row in ds:
        name = row.get("name") or ""
        level = row.get("level") or ""
        abandon_note = row.get("abandon_note") or ""
        levels[level] += 1
        all_names[name] += 1
        if not abandon_note:
            active_names[name] += 1
        if level == "條":
            article_level_names[name] += 1
            if not abandon_note:
                active_article_level_names[name] += 1
        if name in {
            "民法",
            "用戶用電設備裝置規則",
            "民事訴訟法",
            "刑事訴訟法",
            "建築技術規則建築構造編",
            "公司法",
            "船舶設備規則",
            "航空器飛航作業管理規則",
            "中華民國刑法",
            "道路交通管理處罰條例",
            "勞動基準法",
            "銀行法",
            "證券交易法",
        }:
            target_names[name] += 1
        for topic, pattern in rough_topics.items():
            if pattern.search(name):
                rough_topic_counts[topic] += 1

    show("all rows grouped by name", all_names)
    show("active rows only grouped by name", active_names)
    show("level == 條 only grouped by name", article_level_names)
    show("active + level == 條 grouped by name", active_article_level_names)

    print("\n## level values")
    for level, count in levels.most_common():
        print(f"{level or '<empty>'}\t{count}")

    print("\n## selected names, all rows")
    for name, count in target_names.most_common():
        print(f"{name}\t{count}")

    print("\n## rough topic counts by law name keywords, all rows")
    for topic, count in rough_topic_counts.most_common():
        print(f"{topic}\t{count}")


if __name__ == "__main__":
    main()
