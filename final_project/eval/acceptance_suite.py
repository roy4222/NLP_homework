"""Generate and evaluate a 1,000-case legal issue acceptance suite.

The suite is intentionally deterministic and template-based. It is useful for
demo regression checks and presentation evidence, but it should not be described
as an independent human-labeled benchmark.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.model_service import ModelService  # noqa: E402

CASE_PLAN = {
    "typical": 30,
    "colloquial": 10,
    "ambiguous": 5,
    "near_miss": 5,
}

CONTEXTS = [
    "我想知道這種情況要歸到哪一類",
    "朋友建議我先整理法律問題類型",
    "我需要把這件事描述給法律服務窗口",
    "這件事發生在台灣，想先做法律議題分流",
    "請只判斷可能的法律分類",
]

TARGETED_SCENARIOS: dict[str, dict[str, list[str]]] = {
    "dui_public_danger": {
        "typical": [
            "我酒後開車被警察攔下，酒測值超標。",
            "我喝酒後騎機車，路邊臨檢酒精濃度超過標準。",
            "朋友酒駕撞到分隔島，被依公共危險處理。",
        ],
        "colloquial": ["我喝完酒還開車回家，結果被酒測。", "我醉了還騎車，被警察攔。"],
        "ambiguous": ["我只是喝了一點酒後開車，警方說可能是公共危險。"],
        "near_miss": ["不是車禍賠償問題，是酒後開車被酒測超標。"],
    },
    "theft": {
        "typical": ["我的機車停在家門口被偷走。", "同事把我的錢包拿走不還。", "店裡商品被客人偷拿。"],
        "colloquial": ["我的東西不見了，監視器看到有人偷。", "他把我財物拿走。"],
        "ambiguous": ["沒有簽約問題，主要是財物被人拿走。"],
        "near_miss": ["不是詐騙匯款，是有人直接偷走我的手機。"],
    },
    "aggravated_theft": {
        "typical": ["半夜有人撬門侵入住宅偷走財物。", "三個人結夥進倉庫偷東西。", "竊賊夜間破窗進屋。"],
        "colloquial": ["有人半夜闖進我家偷東西。", "他撬門進來拿走電腦。"],
        "ambiguous": ["看起來不是普通偷竊，因為有侵入住宅和夜間行為。"],
        "near_miss": ["不是單純財物被偷，是夜間撬門侵入住宅竊盜。"],
    },
    "fraud": {
        "typical": ["我網購手機匯款後，賣家封鎖我不出貨。", "對方用投資名義騙我匯款。", "我被詐騙集團要求轉帳。"],
        "colloquial": ["我被騙錢了，匯完款人就消失。", "賣家收錢後封鎖我。"],
        "ambiguous": ["不是東西瑕疵，是一開始就騙我匯款。"],
        "near_miss": ["不是不當得利誤匯，是對方用假交易騙我匯款。"],
    },
    "injury": {
        "typical": ["我跟鄰居吵架，被對方揍到瘀青。", "對方毆打我，害我骨折去驗傷。", "朋友被人打傷流血。"],
        "colloquial": ["我被揍好痛，身上有傷。", "他打我，打到我受傷。"],
        "ambiguous": ["不是車禍過失，是對方動手毆打造成傷害。"],
        "near_miss": ["不是過失傷害，是他故意打我受傷。"],
    },
    "negligent_injury": {
        "typical": ["我開車沒注意撞倒路人，對方受傷驗傷。", "車禍中我過失撞傷騎士。", "轉彎時沒注意造成行人受傷。"],
        "colloquial": ["我不小心車禍撞傷人。", "開車沒注意把人撞傷。"],
        "ambiguous": ["不是故意打人，是車禍過失造成受傷。"],
        "near_miss": ["不是肇事逃逸，我有留在現場，但過失撞傷對方。"],
    },
    "negligent_death": {
        "typical": ["我開車過失造成對方死亡。", "工地管理疏失導致工人死亡。", "車禍死亡案件可能涉及過失致死。"],
        "colloquial": ["因為我的疏忽害人死掉。", "車禍讓對方死亡。"],
        "ambiguous": ["不是單純受傷，而是過失行為造成死亡。"],
        "near_miss": ["不是過失傷害，是車禍死亡的過失致死問題。"],
    },
    "hit_and_run": {
        "typical": ["對方撞到我後直接逃走，沒有停下處理。", "車禍肇事後駕駛離開現場。", "司機撞到人後逃逸。"],
        "colloquial": ["他撞了就跑。", "車子撞到我後開車逃走。"],
        "ambiguous": ["不是只有過失撞傷，重點是撞到人後逃逸。"],
        "near_miss": ["不是酒駕問題，是肇事後沒有停下而逃走。"],
    },
    "property_damage": {
        "typical": ["鄰居砸破我的車窗。", "對方故意刮車造成損壞。", "有人破壞我的店門。"],
        "colloquial": ["我的東西被砸壞了。", "他把我車刮壞。"],
        "ambiguous": ["不是偷走財物，是把財物破壞損壞。"],
        "near_miss": ["不是所有物返還，是對方砸壞我的物品。"],
    },
    "document_forgery": {
        "typical": ["有人偽造我的簽名製作文件。", "對方偷蓋我的印章簽約。", "公司文件上的簽名被偽造。"],
        "colloquial": ["那份文件不是我簽的。", "有人冒用我簽名。"],
        "ambiguous": ["不是契約違約，而是簽名文件被偽造。"],
        "near_miss": ["不是單純沒履行合約，是有人偽造文件和印章。"],
    },
    "defamation": {
        "typical": ["有人在社群發文罵我詐騙犯，害我名譽受損。", "對方公開辱罵我。", "網友爆料不實內容毀我名譽。"],
        "colloquial": ["他在網路罵我很難聽。", "有人貼文說我壞話。"],
        "ambiguous": ["不是恐嚇威脅，是公開辱罵造成名譽問題。"],
        "near_miss": ["不是詐欺本身，是別人說我詐欺犯造成妨害名譽。"],
    },
    "intimidation": {
        "typical": ["前男友傳訊息說要讓我好看，叫我小心家人。", "對方威脅要殺我。", "陌生人恐嚇我讓我害怕。"],
        "colloquial": ["他一直威脅我，說要找人處理我。", "我收到恐嚇訊息。"],
        "ambiguous": ["不是單純辱罵，是讓我害怕的人身威脅。"],
        "near_miss": ["不是妨害名譽貼文，是私訊恐嚇要傷害我。"],
    },
    "tort_damages": {
        "typical": ["對方行為造成我損害，我想請求賠償。", "車禍後我想請求精神慰撫金。", "侵權行為造成財產損害。"],
        "colloquial": ["我想跟對方求償。", "我受損了想要賠償。"],
        "ambiguous": ["不確定刑事分類，但民事上想主張損害賠償。"],
        "near_miss": ["不是離婚本身，是侵權行為造成精神損害要慰撫金。"],
    },
    "divorce": {
        "typical": ["配偶外遇多年，我想離婚。", "夫妻長期分居，想聲請裁判離婚。", "婚姻破裂需要離婚諮詢。"],
        "colloquial": ["我想跟另一半離婚。", "配偶外遇，我不想繼續婚姻。"],
        "ambiguous": ["不是單純財產分配，主要是婚姻和離婚問題。"],
        "near_miss": ["不是妨害名譽，是配偶外遇導致離婚。"],
    },
    "unjust_enrichment": {
        "typical": ["我匯錯錢到陌生人帳戶，對方拒絕返還。", "店家多收款項卻不退。", "對方無法律上原因取得我的錢。"],
        "colloquial": ["我把錢匯錯了，對方不還。", "他多拿我的錢不返還。"],
        "ambiguous": ["不是詐騙，是誤匯後對方得利不還。"],
        "near_miss": ["不是假交易騙錢，是匯錯帳戶後拒絕把錢還給我。"],
    },
    "contract_breach": {
        "typical": ["廠商簽約收訂金後沒有依約完成工作。", "對方違約不履行契約。", "合約約定期限到了仍遲延交付。"],
        "colloquial": ["收了訂金卻沒做事。", "簽約後對方沒履行。"],
        "ambiguous": ["不是詐騙匯款，是契約成立後債務不履行。"],
        "near_miss": ["不是買賣瑕疵，是廠商根本沒有依約完成工作。"],
    },
    "sales_defect": {
        "typical": ["我買二手車後發現引擎有重大問題，賣家拒絕修理。", "交車後發現商品有瑕疵想退貨。", "產品保固內壞掉，店家不處理。"],
        "colloquial": ["東西買回來才發現壞掉。", "賣家不給退錢也不修理。"],
        "ambiguous": ["不是沒交貨，而是交付物本身有瑕疵。"],
        "near_miss": ["不是契約完全沒履行，是買賣標的有重大瑕疵。"],
    },
    "inheritance": {
        "typical": ["父親過世後遺產要如何由繼承人分配。", "家人對遺囑內容有爭議。", "叔伯和子女爭遺產。"],
        "colloquial": ["長輩過世留下財產，不知道誰能繼承。", "我們家在吵遺產。"],
        "ambiguous": ["不是婚後財產，是死亡後遺產和繼承人問題。"],
        "near_miss": ["不是夫妻財產分配，是父母過世後的繼承問題。"],
    },
    "marital_property": {
        "typical": ["離婚時想請求夫妻剩餘財產分配。", "婚後財產要怎麼分配。", "配偶名下財產很多，離婚財產分配有爭議。"],
        "colloquial": ["離婚後財產要怎麼分。", "婚後存款算誰的。"],
        "ambiguous": ["不是是否離婚，而是離婚後夫妻財產分配。"],
        "near_miss": ["不是繼承遺產，是夫妻剩餘財產分配。"],
    },
    "property_return": {
        "typical": ["朋友借用我的車不還，我想請求返還。", "對方占有我的物品拒絕歸還。", "我的所有物被拿走，想拿回。"],
        "colloquial": ["東西借出去拿不回來。", "他占有我的東西不還。"],
        "ambiguous": ["不是毀損破壞，是所有物返還請求。"],
        "near_miss": ["不是偷竊刑事問題，我主要想把借用不還的物品拿回。"],
    },
}


def cycle(items: list[str], count: int) -> Iterable[str]:
    for index in range(count):
        yield items[index % len(items)]


def generate_cases(issues: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for issue in issues:
        issue_id = issue["id"]
        scenarios = TARGETED_SCENARIOS[issue_id]
        for split, count in CASE_PLAN.items():
            for offset, text in enumerate(cycle(scenarios[split], count), start=1):
                context = CONTEXTS[(offset - 1) % len(CONTEXTS)]
                cases.append(
                    {
                        "id": f"{issue_id}-{split}-{offset:02d}",
                        "label": issue_id,
                        "label_zh": issue["name_zh"],
                        "split": split,
                        "text": f"{text} {context}。",
                    }
                )
    return cases


def predict_case(service: ModelService, case: dict) -> dict:
    result = service.predict(case["text"])
    predictions = [row["id"] for row in result["final_prediction"]]
    return {
        **case,
        "predictions": predictions,
        "top1": predictions[0] if predictions else None,
        "top3": predictions[:3],
        "matched_keywords": result["explanation"]["matched_keywords"],
    }


def compute_acceptance_metrics(rows: list[dict], labels: list[str]) -> dict:
    per_label: dict[str, dict] = {}
    predicted_counts = Counter(pred for row in rows for pred in row["predictions"])
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        top1_hits = sum(row["top1"] == label for row in label_rows)
        top3_hits = sum(label in row["top3"] for row in label_rows)
        recall_hits = sum(label in row["predictions"] for row in label_rows)
        precision_denominator = predicted_counts[label]
        precision = recall_hits / precision_denominator if precision_denominator else 0.0
        per_label[label] = {
            "cases": len(label_rows),
            "top1_accuracy": top1_hits / len(label_rows),
            "top3_hit_rate": top3_hits / len(label_rows),
            "recall": recall_hits / len(label_rows),
            "precision": precision,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "recall_hits": recall_hits,
            "predicted_count": precision_denominator,
            "passed": (
                top1_hits / len(label_rows) >= 0.95
                and recall_hits / len(label_rows) >= 0.95
                and precision >= 0.90
            ),
        }

    split_metrics = {}
    for split in CASE_PLAN:
        split_rows = [row for row in rows if row["split"] == split]
        split_metrics[split] = {
            "cases": len(split_rows),
            "top1_accuracy": sum(row["top1"] == row["label"] for row in split_rows) / len(split_rows),
            "top3_hit_rate": sum(row["label"] in row["top3"] for row in split_rows) / len(split_rows),
            "recall": sum(row["label"] in row["predictions"] for row in split_rows) / len(split_rows),
        }

    return {
        "total_cases": len(rows),
        "label_count": len(labels),
        "passed_labels": sum(item["passed"] for item in per_label.values()),
        "top1_accuracy": sum(row["top1"] == row["label"] for row in rows) / len(rows),
        "top3_hit_rate": sum(row["label"] in row["top3"] for row in rows) / len(rows),
        "recall": sum(row["label"] in row["predictions"] for row in rows) / len(rows),
        "per_label": per_label,
        "by_split": split_metrics,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(metrics: dict, path: Path) -> None:
    lines = [
        "# Acceptance Test Results",
        "",
        "This acceptance suite is a deterministic, template-based demo regression set.",
        "It contains 50 cases for each of the 20 labels (1,000 total), but each label's",
        "cases are cycled from only ~3-5 base phrasings across 4 difficulty splits, so the",
        "linguistic diversity is far lower than the case count suggests.",
        "It should be treated as course-demo validation, not as an independent human-labeled benchmark.",
        "",
        "## Summary",
        "",
        f"- Total cases: {metrics['total_cases']}",
        f"- Labels tested: {metrics['label_count']}",
        f"- Labels passing threshold: {metrics['passed_labels']} / {metrics['label_count']}",
        f"- Overall Top-1 accuracy: {metrics['top1_accuracy']:.3f}",
        f"- Overall Top-3 hit rate: {metrics['top3_hit_rate']:.3f}",
        f"- Overall recall: {metrics['recall']:.3f}",
        "",
        "Threshold: per-label Top-1 accuracy >= 0.95, recall >= 0.95, precision >= 0.90.",
        "",
        "## By Scenario Type",
        "",
        "Note: predictions are the merged top-5, so \"Recall\" below is recall@5.",
        "",
        "| Split | Cases | Top-1 Accuracy | Top-3 Hit Rate | Recall@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in metrics["by_split"].items():
        lines.append(
            f"| {split} | {row['cases']} | {row['top1_accuracy']:.3f} | "
            f"{row['top3_hit_rate']:.3f} | {row['recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-label Results",
            "",
            "| Label | Cases | Top-1 | Top-3 | Recall@5 | Precision | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in metrics["per_label"].items():
        lines.append(
            f"| `{label}` | {row['cases']} | {row['top1_accuracy']:.3f} | "
            f"{row['top3_hit_rate']:.3f} | {row['recall']:.3f} | "
            f"{row['precision']:.3f} | {'yes' if row['passed'] else 'no'} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ontology_path = Path("data/issues.yaml")
    issues = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    labels = [issue["id"] for issue in issues]
    cases = generate_cases(issues)
    service = ModelService(ontology_path=ontology_path)
    predictions = [predict_case(service, case) for case in cases]
    metrics = compute_acceptance_metrics(predictions, labels)

    write_jsonl(cases, Path("data/acceptance_cases.jsonl"))
    Path("data/acceptance_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("data/acceptance_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics, Path("report/acceptance_results.md"))

    print(
        "acceptance "
        f"cases={metrics['total_cases']} "
        f"top1={metrics['top1_accuracy']:.3f} "
        f"top3={metrics['top3_hit_rate']:.3f} "
        f"recall={metrics['recall']:.3f} "
        f"passed_labels={metrics['passed_labels']}/{metrics['label_count']}"
    )


if __name__ == "__main__":
    main()
