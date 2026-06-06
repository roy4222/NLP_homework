# Final Project Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Chinese NLP final project deliverable: a legal issue triage classifier, an interactive Next.js demo backed by Flask, a reproducible experiment pipeline, slides, and a Markdown report that satisfies the course guidelines.

**Architecture:** The project separates model/data work from the demo UI. Python scripts build weak labels from `tw-legal-synthetic-qa`, use `tw-processed-law-article` for citation lookup and ontology construction, train/evaluate rule-based and TF-IDF/SVM models, then expose predictions through a Flask API. A Next.js + shadcn/ui frontend calls the API and visualizes predictions, model comparisons, explanations, and limitations without generating legal advice.

**Tech Stack:** Python 3.12, uv, datasets, pandas, scikit-learn, jieba, Flask, pytest, Next.js, TypeScript, Tailwind CSS, shadcn/ui, lucide-react, Markdown.

---

## File Structure

- Create: `final_project/README.md` - runnable project overview, setup, commands, dataset references.
- Create: `final_project/pyproject.toml` - Python dependencies for data, models, API, tests.
- Create: `final_project/data/issues.yaml` - MVP legal issue ontology, mapping issue labels to law/article pairs, aliases, keywords.
- Create: `final_project/scripts/citation_normalizer.py` - normalize legal citations from assistant answers.
- Create: `final_project/scripts/build_labels.py` - convert synthetic QA rows into weakly labeled JSONL.
- Create: `final_project/scripts/split_data.py` - deterministic train/val/test split and stats.
- Create: `final_project/baselines/rule.py` - keyword/alias baseline.
- Create: `final_project/baselines/tfidf_svm.py` - TF-IDF + One-vs-Rest classifier.
- Create: `final_project/eval/metrics.py` - multi-label metrics and report export.
- Create: `final_project/api/app.py` - Flask API with `/api/health` and `/api/predict`.
- Create: `final_project/api/model_service.py` - load ontology and model outputs; supports mock mode before models exist.
- Create: `final_project/web/` - Next.js frontend using shadcn/ui.
- Create: `final_project/report/report.md` - final written report in English, matching course guideline sections.
- Create: `final_project/report/slides.md` - slide outline and English speaker notes for 8-10 minutes.
- Create: `final_project/report/references.md` - citations for datasets, models, and competitor products.
- Modify: `final_project/FINAL_PROJECT_SPEC.md` - keep as source-of-truth spec when decisions change.
- Modify: root `.gitignore` - ignore raw datasets, model checkpoints, frontend build artifacts.

---

## Task 1: Bootstrap Project Dependencies

**Files:**
- Create: `final_project/README.md`
- Create: `final_project/pyproject.toml`
- Create: `final_project/.gitignore`
- Modify: `.gitignore`

- [ ] **Step 1: Create project README**

Create `final_project/README.md`:

```markdown
# Chinese Legal Issue Triage

This final project builds a Chinese NLP classifier for legal issue triage.
It is not a legal advice chatbot and does not generate legal opinions.

## Task

Input: an everyday Chinese legal scenario.

Output: one or more legal issue labels, confidence scores, and supporting law articles as lookup clues.

## Main Datasets

- `lianghsun/tw-legal-synthetic-qa`: main weak-label source.
- `lianghsun/tw-processed-law-article`: statute lookup and ontology support.

## Methods

1. Citation normalization and weak labeling
2. Rule-based baseline
3. TF-IDF + One-vs-Rest SVM / Logistic Regression
4. Optional BERT fine-tuning

## Demo

The interactive demo uses a Flask API and a Next.js + shadcn/ui frontend.

## Disclaimer

This project performs legal issue triage for NLP education. It does not provide legal advice.
```

- [ ] **Step 2: Create Python dependency file**

Create `final_project/pyproject.toml`:

```toml
[project]
name = "legal-issue-triage"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "datasets>=4.0",
    "flask>=3.0",
    "flask-cors>=4.0",
    "jieba>=0.42",
    "joblib>=1.4",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.5",
    "seaborn>=0.13",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]
bert = ["torch>=2.2", "transformers>=4.40", "accelerate>=0.30"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create final project gitignore**

Create `final_project/.gitignore`:

```gitignore
data/raw/
data/hf_cache/
data/*.joblib
models/**/checkpoint-*/
models/**/*.pt
models/**/*.bin
web/.next/
web/node_modules/
web/out/
.env
.hf_token
```

- [ ] **Step 4: Update root gitignore**

Append to `.gitignore` if not already present:

```gitignore
# final project artifacts
final_project/data/raw/
final_project/data/hf_cache/
final_project/models/**/checkpoint-*/
final_project/models/**/*.pt
final_project/models/**/*.bin
final_project/web/.next/
final_project/web/node_modules/
final_project/web/out/
```

- [ ] **Step 5: Verify Python dependencies install**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv sync --extra dev
```

Expected: dependency resolution succeeds.

- [ ] **Step 6: Commit**

```bash
cd /home/roy422/NLP_homework
git add .gitignore final_project/README.md final_project/pyproject.toml final_project/.gitignore
git commit -m "chore(final): bootstrap final project dependencies"
```

---

## Task 2: Build Citation Normalization

**Files:**
- Create: `final_project/scripts/citation_normalizer.py`
- Create: `final_project/tests/test_citation_normalizer.py`

- [ ] **Step 1: Write failing tests**

Create `final_project/tests/test_citation_normalizer.py`:

```python
from scripts.citation_normalizer import extract_citations


def test_extract_dui_article_with_subarticle():
    text = "依中華民國刑法第185條之3，酒駕可能構成公共危險罪。"
    assert extract_citations(text) == {("刑法", "185-3")}


def test_extract_chinese_numeral():
    text = "依民法第一百八十四條，侵權行為應負損害賠償責任。"
    assert extract_citations(text) == {("民法", "184")}


def test_normalize_tai_variant_and_spaces():
    text = "臺灣實務常引用 刑事訴訟法 第 四百四十九 條。"
    assert extract_citations(text) == {("刑事訴訟法", "449")}


def test_deduplicate_repeated_article():
    text = "刑法第320條是竊盜。刑法第320條也可能搭配其他條文。"
    assert extract_citations(text) == {("刑法", "320")}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run pytest tests/test_citation_normalizer.py -v
```

Expected: FAIL because `scripts.citation_normalizer` does not exist.

- [ ] **Step 3: Implement citation normalizer**

Create `final_project/scripts/citation_normalizer.py`:

```python
"""Normalize Taiwan legal citations into stable (law_name, article_key) pairs."""

from __future__ import annotations

import re
import unicodedata

LAW_ALIASES = {
    "中華民國刑法": "刑法",
    "中華民國民法": "民法",
    "中華民國刑事訴訟法": "刑事訴訟法",
    "臺灣刑法": "刑法",
    "台灣刑法": "刑法",
}

LAW_NAMES = sorted(
    {
        "中華民國刑法",
        "中華民國民法",
        "中華民國刑事訴訟法",
        "刑事訴訟法",
        "民事訴訟法",
        "道路交通管理處罰條例",
        "毒品危害防制條例",
        "家庭暴力防治法",
        "勞動基準法",
        "刑法",
        "民法",
    },
    key=len,
    reverse=True,
)

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
    "(" + "|".join(re.escape(name) for name in LAW_NAMES) + ")"
    r"\s*第\s*([0-9零一二三四五六七八九十百千]+)\s*條"
    r"(?:\s*之\s*([0-9零一二三四五六七八九十百千]+))?"
)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("臺", "台")


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
    return LAW_ALIASES.get(law_name, law_name)


def article_key(main: str, sub: str | None) -> str:
    main_key = str(cn_to_int(main))
    if sub:
        return f"{main_key}-{cn_to_int(sub)}"
    return main_key


def extract_citations(text: str) -> set[tuple[str, str]]:
    normalized = normalize_text(text)
    pairs: set[tuple[str, str]] = set()
    for law_name, main, sub in CITATION_RE.findall(normalized):
        pairs.add((normalize_law_name(law_name), article_key(main, sub or None)))
    return pairs
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run pytest tests/test_citation_normalizer.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/scripts/citation_normalizer.py final_project/tests/test_citation_normalizer.py
git commit -m "feat(final): add legal citation normalizer"
```

---

## Task 3: Create MVP Ontology

**Files:**
- Create: `final_project/data/issues.yaml`
- Create: `final_project/tests/test_issues_yaml.py`

- [ ] **Step 1: Create ontology validation test**

Create `final_project/tests/test_issues_yaml.py`:

```python
from pathlib import Path

import yaml


def test_issues_yaml_has_valid_minimum_shape():
    issues = yaml.safe_load(Path("data/issues.yaml").read_text(encoding="utf-8"))
    assert 15 <= len(issues) <= 25
    ids = [issue["id"] for issue in issues]
    assert len(ids) == len(set(ids))
    for issue in issues:
        assert issue["id"]
        assert issue["name_zh"]
        assert issue["name_en"]
        assert issue["category"] in {"criminal", "civil"}
        assert issue["laws"]
        assert issue["aliases"]
        for law in issue["laws"]:
            assert set(law) == {"law", "article"}
```

- [ ] **Step 2: Create MVP ontology**

Create `final_project/data/issues.yaml`:

```yaml
- id: dui_public_danger
  name_zh: 酒駕 / 公共危險
  name_en: DUI / public danger
  category: criminal
  laws:
    - law: 刑法
      article: "185-3"
  aliases: [酒駕, 酒後駕車, 不能安全駕駛, 公共危險]
  keywords: [酒, 開車, 駕駛, 酒測, 吐氣, 警察攔下]

- id: theft
  name_zh: 竊盜
  name_en: theft
  category: criminal
  laws:
    - law: 刑法
      article: "320"
  aliases: [偷竊, 竊盜, 偷東西]
  keywords: [偷, 竊, 拿走, 未經同意]

- id: aggravated_theft
  name_zh: 加重竊盜
  name_en: aggravated theft
  category: criminal
  laws:
    - law: 刑法
      article: "321"
  aliases: [加重竊盜, 侵入住宅竊盜, 結夥竊盜]
  keywords: [侵入, 住宅, 結夥, 攜帶兇器]

- id: fraud
  name_zh: 詐欺
  name_en: fraud
  category: criminal
  laws:
    - law: 刑法
      article: "339"
  aliases: [詐欺, 被騙, 騙錢]
  keywords: [騙, 詐, 匯款, 投資, 假冒]

- id: injury
  name_zh: 傷害
  name_en: injury
  category: criminal
  laws:
    - law: 刑法
      article: "277"
  aliases: [傷害, 打人, 毆打]
  keywords: [打, 推, 傷, 流血, 骨折]

- id: negligent_injury
  name_zh: 過失傷害
  name_en: negligent injury
  category: criminal
  laws:
    - law: 刑法
      article: "284"
  aliases: [過失傷害, 車禍受傷]
  keywords: [過失, 車禍, 撞到, 受傷]

- id: negligent_death
  name_zh: 過失致死
  name_en: negligent homicide
  category: criminal
  laws:
    - law: 刑法
      article: "276"
  aliases: [過失致死, 過失死亡]
  keywords: [死亡, 致死, 過失]

- id: hit_and_run
  name_zh: 肇事逃逸
  name_en: hit and run
  category: criminal
  laws:
    - law: 刑法
      article: "185-4"
  aliases: [肇事逃逸, 撞人逃跑]
  keywords: [逃逸, 跑掉, 沒有停下, 肇事]

- id: property_damage
  name_zh: 毀損
  name_en: property damage
  category: criminal
  laws:
    - law: 刑法
      article: "354"
  aliases: [毀損, 破壞財物]
  keywords: [砸, 毀, 破壞, 刮傷]

- id: document_forgery
  name_zh: 偽造文書
  name_en: document forgery
  category: criminal
  laws:
    - law: 刑法
      article: "210"
  aliases: [偽造文書, 假文件, 冒簽]
  keywords: [偽造, 文書, 簽名, 文件]

- id: defamation
  name_zh: 妨害名譽
  name_en: defamation
  category: criminal
  laws:
    - law: 刑法
      article: "309"
    - law: 刑法
      article: "310"
    - law: 民法
      article: "195"
  aliases: [妨害名譽, 誹謗, 公然侮辱]
  keywords: [罵, 誹謗, 侮辱, 名譽, 貼文]

- id: intimidation
  name_zh: 恐嚇
  name_en: intimidation
  category: criminal
  laws:
    - law: 刑法
      article: "305"
  aliases: [恐嚇, 威脅]
  keywords: [威脅, 恐嚇, 要殺, 要打]

- id: tort_damages
  name_zh: 侵權行為 / 損害賠償
  name_en: tort damages
  category: civil
  laws:
    - law: 民法
      article: "184"
    - law: 民法
      article: "193"
    - law: 民法
      article: "195"
  aliases: [侵權行為, 損害賠償, 賠償]
  keywords: [賠償, 損害, 侵權, 醫藥費, 精神慰撫金]

- id: divorce
  name_zh: 離婚
  name_en: divorce
  category: civil
  laws:
    - law: 民法
      article: "1052"
  aliases: [離婚, 裁判離婚]
  keywords: [離婚, 配偶, 外遇, 婚姻]

- id: unjust_enrichment
  name_zh: 不當得利
  name_en: unjust enrichment
  category: civil
  laws:
    - law: 民法
      article: "179"
  aliases: [不當得利, 沒有法律原因]
  keywords: [返還, 匯錯, 沒有原因, 受利益]

- id: contract_breach
  name_zh: 契約責任 / 債務不履行
  name_en: contract breach
  category: civil
  laws:
    - law: 民法
      article: "226"
    - law: 民法
      article: "227"
  aliases: [債務不履行, 不完全給付, 契約違約]
  keywords: [契約, 合約, 違約, 沒履行, 延遲]

- id: sales_defect
  name_zh: 買賣瑕疵
  name_en: sales defect
  category: civil
  laws:
    - law: 民法
      article: "354"
    - law: 民法
      article: "359"
    - law: 民法
      article: "360"
  aliases: [瑕疵擔保, 買賣瑕疵]
  keywords: [瑕疵, 退貨, 減價, 商品, 買賣]

- id: inheritance
  name_zh: 繼承
  name_en: inheritance
  category: civil
  laws:
    - law: 民法
      article: "1138"
  aliases: [繼承, 繼承人]
  keywords: [遺產, 繼承, 父母過世, 兄弟姐妹]

- id: marital_property
  name_zh: 夫妻財產
  name_en: marital property
  category: civil
  laws:
    - law: 民法
      article: "1030-1"
  aliases: [夫妻剩餘財產分配, 夫妻財產]
  keywords: [剩餘財產, 婚後財產, 配偶財產]

- id: property_return
  name_zh: 所有物返還 / 物權請求
  name_en: property return
  category: civil
  laws:
    - law: 民法
      article: "767"
  aliases: [所有物返還, 物上請求權]
  keywords: [返還, 占有, 所有權, 土地, 房屋]
```

- [ ] **Step 3: Run ontology validation**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run pytest tests/test_issues_yaml.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/data/issues.yaml final_project/tests/test_issues_yaml.py
git commit -m "feat(final): define MVP legal issue ontology"
```

---

## Task 4: Build Weak Labels and Data Splits

**Files:**
- Create: `final_project/scripts/build_labels.py`
- Create: `final_project/scripts/split_data.py`
- Generated: `final_project/data/labeled.jsonl`
- Generated: `final_project/data/train.jsonl`
- Generated: `final_project/data/val.jsonl`
- Generated: `final_project/data/test.jsonl`
- Generated: `final_project/data/stats.md`

- [ ] **Step 1: Implement weak label builder**

Create `final_project/scripts/build_labels.py`:

```python
"""Build weak labels from tw-legal-synthetic-qa using the issue ontology."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset
from scripts.citation_normalizer import extract_citations


def load_issue_map(path: Path) -> dict[tuple[str, str], list[str]]:
    issues = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], list[str]] = {}
    for issue in issues:
        for law in issue["laws"]:
            mapping.setdefault((law["law"], str(law["article"])), []).append(issue["id"])
    return mapping


def get_message(row: dict, role: str) -> str:
    for message in row["messages"]:
        if message.get("role") == role:
            return message.get("content", "")
    return ""


def main() -> None:
    issue_map = load_issue_map(Path("data/issues.yaml"))
    ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
    rows = list(ds["train"]) + list(ds["test"])
    output = Path("data/labeled.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    label_counter: Counter[str] = Counter()
    kept = 0
    discarded = 0

    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            story = get_message(row, "user").strip()
            analysis = get_message(row, "assistant").strip()
            citations = sorted(extract_citations(analysis))
            labels = sorted({label for pair in citations for label in issue_map.get(pair, [])})
            if not story or len(story) < 20 or not labels:
                discarded += 1
                continue
            kept += 1
            label_counter.update(labels)
            item = {
                "id": f"synthetic-{idx}",
                "story": story,
                "labels": labels,
                "citations": citations,
                "label_quality": "usable",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"kept={kept}")
    print(f"discarded={discarded}")
    for label, count in label_counter.most_common():
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run weak label builder**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python scripts/build_labels.py
```

Expected: `data/labeled.jsonl` exists and prints label distribution.

- [ ] **Step 3: Implement data splitter**

Create `final_project/scripts/split_data.py`:

```python
"""Split labeled data into train/val/test and write dataset statistics."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def label_counts(rows: list[dict]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row["labels"])
    return counter


def main() -> None:
    rows = read_jsonl(Path("data/labeled.jsonl"))
    random.Random(42).shuffle(rows)
    n = len(rows)
    train = rows[: int(n * 0.7)]
    val = rows[int(n * 0.7) : int(n * 0.85)]
    test = rows[int(n * 0.85) :]

    write_jsonl(Path("data/train.jsonl"), train)
    write_jsonl(Path("data/val.jsonl"), val)
    write_jsonl(Path("data/test.jsonl"), test)

    stats = [
        "# Dataset Statistics",
        "",
        f"- Total labeled rows: {n}",
        f"- Train: {len(train)}",
        f"- Validation: {len(val)}",
        f"- Test: {len(test)}",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in label_counts(rows).most_common():
        stats.append(f"| `{label}` | {count} |")
    Path("data/stats.md").write_text("\n".join(stats) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run data splitter**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python scripts/split_data.py
```

Expected: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, and `data/stats.md` exist.

- [ ] **Step 5: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/scripts/build_labels.py final_project/scripts/split_data.py final_project/data/labeled.jsonl final_project/data/train.jsonl final_project/data/val.jsonl final_project/data/test.jsonl final_project/data/stats.md
git commit -m "feat(final): build weak labels and data splits"
```

---

## Task 5: Train Rule-Based and TF-IDF/SVM Models

**Files:**
- Create: `final_project/baselines/rule.py`
- Create: `final_project/baselines/tfidf_svm.py`
- Generated: `final_project/data/rule_predictions.json`
- Generated: `final_project/data/tfidf_svm.joblib`
- Generated: `final_project/data/tfidf_svm_predictions.json`

- [ ] **Step 1: Implement rule-based baseline**

Create `final_project/baselines/rule.py`:

```python
"""Rule-based baseline using ontology aliases and keywords."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def predict_one(story: str, issues: list[dict]) -> list[dict]:
    predictions = []
    for issue in issues:
        terms = issue.get("aliases", []) + issue.get("keywords", [])
        hits = [term for term in terms if term and term in story]
        if hits:
            predictions.append(
                {
                    "id": issue["id"],
                    "name_zh": issue["name_zh"],
                    "confidence": min(0.95, 0.45 + 0.1 * len(hits)),
                    "evidence": hits[:5],
                }
            )
    return sorted(predictions, key=lambda item: item["confidence"], reverse=True)


def main() -> None:
    issues = yaml.safe_load(Path("data/issues.yaml").read_text(encoding="utf-8"))
    rows = read_jsonl(Path("data/test.jsonl"))
    output = []
    for row in rows:
        output.append(
            {
                "id": row["id"],
                "gold": row["labels"],
                "predicted": [item["id"] for item in predict_one(row["story"], issues)],
                "details": predict_one(row["story"], issues),
            }
        )
    Path("data/rule_predictions.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run rule baseline**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python baselines/rule.py
```

Expected: `data/rule_predictions.json` exists.

- [ ] **Step 3: Implement TF-IDF/SVM baseline**

Create `final_project/baselines/tfidf_svm.py`:

```python
"""TF-IDF + One-vs-Rest Linear SVM for multi-label issue classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    train = read_jsonl(Path("data/train.jsonl"))
    test = read_jsonl(Path("data/test.jsonl"))
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([row["labels"] for row in train])

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2)),
            ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced"))),
        ]
    )
    pipeline.fit([row["story"] for row in train], y_train)

    y_pred = pipeline.predict([row["story"] for row in test])
    labels = mlb.inverse_transform(y_pred)
    predictions = [
        {"id": row["id"], "gold": row["labels"], "predicted": sorted(list(pred))}
        for row, pred in zip(test, labels)
    ]

    joblib.dump({"pipeline": pipeline, "mlb": mlb}, "data/tfidf_svm.joblib")
    Path("data/tfidf_svm_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run TF-IDF/SVM baseline**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python baselines/tfidf_svm.py
```

Expected: `data/tfidf_svm.joblib` and `data/tfidf_svm_predictions.json` exist.

- [ ] **Step 5: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/baselines/rule.py final_project/baselines/tfidf_svm.py final_project/data/rule_predictions.json final_project/data/tfidf_svm.joblib final_project/data/tfidf_svm_predictions.json
git commit -m "feat(final): train rule and tfidf svm baselines"
```

---

## Task 6: Evaluate Models and Export Results

**Files:**
- Create: `final_project/eval/metrics.py`
- Generated: `final_project/report/results.md`

- [ ] **Step 1: Implement metrics script**

Create `final_project/eval/metrics.py`:

```python
"""Evaluate multi-label predictions and export Markdown result tables."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_predictions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score(name: str, rows: list[dict]) -> dict:
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform([row["gold"] for row in rows])
    y_pred = mlb.transform([row["predicted"] for row in rows])
    return {
        "method": name,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def main() -> None:
    results = [
        score("Rule-based", load_predictions(Path("data/rule_predictions.json"))),
        score("TF-IDF + SVM", load_predictions(Path("data/tfidf_svm_predictions.json"))),
    ]
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
    Path("report").mkdir(exist_ok=True)
    Path("report/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run metrics**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python eval/metrics.py
```

Expected: `report/results.md` contains a Markdown metrics table.

- [ ] **Step 3: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/eval/metrics.py final_project/report/results.md
git commit -m "feat(final): export classifier evaluation metrics"
```

---

## Task 7: Build Flask Prediction API

**Files:**
- Create: `final_project/api/app.py`
- Create: `final_project/api/model_service.py`
- Create: `final_project/tests/test_api.py`

- [ ] **Step 1: Write API tests**

Create `final_project/tests/test_api.py`:

```python
from api.app import create_app


def test_health_endpoint():
    client = create_app().test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_predict_endpoint_returns_labels():
    client = create_app().test_client()
    response = client.post("/api/predict", json={"text": "我酒後開車被警察攔下"})
    assert response.status_code == 200
    assert response.json["input"]
    assert response.json["final_prediction"]
    assert response.json["disclaimer"]
```

- [ ] **Step 2: Implement model service**

Create `final_project/api/model_service.py`:

```python
"""Prediction service for the Flask API."""

from __future__ import annotations

from pathlib import Path

import yaml


class ModelService:
    def __init__(self, ontology_path: Path = Path("data/issues.yaml")) -> None:
        self.issues = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))

    def predict(self, text: str) -> dict:
        predictions = []
        for issue in self.issues:
            terms = issue.get("aliases", []) + issue.get("keywords", [])
            hits = [term for term in terms if term and term in text]
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
        predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
        needs_review = not predictions or predictions[0]["confidence"] < 0.6
        return {
            "input": text,
            "status": "ok",
            "final_prediction": predictions[:5],
            "models": {
                "rule": predictions[:5],
                "tfidf_svm": [],
                "bert": [],
            },
            "explanation": {
                "matched_keywords": sorted({hit for pred in predictions for hit in pred["evidence"]}),
                "needs_review": needs_review,
            },
            "disclaimer": "This tool provides legal issue triage for NLP demonstration only. It is not legal advice.",
        }
```

- [ ] **Step 3: Implement Flask app**

Create `final_project/api/app.py`:

```python
"""Flask API for the legal issue triage demo."""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from api.model_service import ModelService


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    service = ModelService()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if len(text) < 5:
            return jsonify({"status": "error", "message": "Please enter a longer Chinese legal scenario."}), 400
        return jsonify(service.predict(text))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
```

- [ ] **Step 4: Run API tests**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run pytest tests/test_api.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/api final_project/tests/test_api.py
git commit -m "feat(final): add flask prediction api"
```

---

## Task 8: Build Next.js shadcn/ui Demo

**Files:**
- Create: `final_project/web/package.json`
- Create: `final_project/web/app/page.tsx`
- Create: `final_project/web/app/layout.tsx`
- Create: `final_project/web/components/ScenarioInput.tsx`
- Create: `final_project/web/components/PredictionResults.tsx`
- Create: `final_project/web/lib/api.ts`

- [ ] **Step 1: Scaffold Next.js app**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
npx create-next-app@latest web --typescript --tailwind --eslint --app --src-dir false --import-alias "@/*"
cd web
npx shadcn@latest init
npx shadcn@latest add button textarea card badge tabs progress alert separator tooltip scroll-area skeleton table
npm install lucide-react
```

Expected: `final_project/web/package.json` exists and shadcn components are installed.

- [ ] **Step 2: Add frontend API helper**

Create `final_project/web/lib/api.ts`:

```typescript
export type Prediction = {
  id: string;
  name_zh: string;
  name_en: string;
  confidence: number;
  supporting_laws: { law: string; article: string }[];
  evidence: string[];
};

export type PredictResponse = {
  input: string;
  status: string;
  final_prediction: Prediction[];
  models: Record<string, Prediction[]>;
  explanation: {
    matched_keywords: string[];
    needs_review: boolean;
  };
  disclaimer: string;
};

export async function predictLegalIssues(text: string): Promise<PredictResponse> {
  const response = await fetch("http://127.0.0.1:5000/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.message || "Prediction failed");
  }

  return response.json();
}
```

- [ ] **Step 3: Add input component**

Create `final_project/web/components/ScenarioInput.tsx`:

```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

const examples = [
  "我酒後開車被警察攔下，酒測超標，想知道可能涉及什麼法律問題。",
  "有人在 Threads 上冒用我的照片發文罵人，現在對方說要告我妨害名譽。",
  "我買到的二手車交車後才發現引擎有重大問題，賣家不願意處理。",
];

type Props = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
};

export function ScenarioInput({ value, loading, onChange, onAnalyze }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Chinese Legal Scenario</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="min-h-44 resize-none"
          placeholder="請輸入一段中文法律情境..."
        />
        <div className="flex flex-wrap gap-2">
          {examples.map((example) => (
            <Button key={example} type="button" variant="outline" size="sm" onClick={() => onChange(example)}>
              Load example
            </Button>
          ))}
        </div>
        <div className="flex gap-2">
          <Button type="button" onClick={onAnalyze} disabled={loading || value.trim().length < 5}>
            {loading ? "Analyzing..." : "Analyze"}
          </Button>
          <Button type="button" variant="ghost" onClick={() => onChange("")}>
            Clear
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 4: Add result component**

Create `final_project/web/components/PredictionResults.tsx`:

```tsx
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { PredictResponse } from "@/lib/api";

type Props = {
  result: PredictResponse | null;
  error: string | null;
};

export function PredictionResults({ result, error }: Props) {
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Prediction failed</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Enter a scenario and run analysis to see predicted legal issue labels.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {result.explanation.needs_review && (
        <Alert>
          <AlertTitle>Needs review</AlertTitle>
          <AlertDescription>The scenario may be too short or ambiguous. Treat labels as tentative.</AlertDescription>
        </Alert>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Predicted Legal Issues</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {result.final_prediction.map((prediction) => (
            <div key={prediction.id} className="rounded-lg border p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="font-medium">{prediction.name_zh}</div>
                  <div className="text-sm text-muted-foreground">{prediction.name_en}</div>
                </div>
                <Badge variant="secondary">{Math.round(prediction.confidence * 100)}%</Badge>
              </div>
              <Progress value={prediction.confidence * 100} className="mt-3" />
              <div className="mt-3 flex flex-wrap gap-2">
                {prediction.supporting_laws.map((law) => (
                  <Badge key={`${law.law}-${law.article}`} variant="outline">
                    {law.law} §{law.article}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
          <p className="text-xs text-muted-foreground">{result.disclaimer}</p>
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 5: Add main page**

Replace `final_project/web/app/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { ScenarioInput } from "@/components/ScenarioInput";
import { PredictionResults } from "@/components/PredictionResults";
import { predictLegalIssues, type PredictResponse } from "@/lib/api";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function analyze() {
    setLoading(true);
    setError(null);
    try {
      setResult(await predictLegalIssues(text));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-dvh bg-zinc-50 p-4 text-zinc-950 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="space-y-2">
          <p className="text-sm font-medium text-teal-700">NLP Final Project Demo</p>
          <h1 className="text-3xl font-semibold tracking-normal">Chinese Legal Issue Triage</h1>
          <p className="max-w-3xl text-muted-foreground">
            Multi-label classification for everyday Chinese legal scenarios. This demo visualizes issue labels,
            supporting articles, and uncertainty. It does not provide legal advice.
          </p>
        </header>
        <section className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <ScenarioInput value={text} loading={loading} onChange={setText} onAnalyze={analyze} />
          <PredictionResults result={result} error={error} />
        </section>
      </div>
    </main>
  );
}
```

- [ ] **Step 6: Run frontend checks**

Run:

```bash
cd /home/roy422/NLP_homework/final_project/web
npm run lint
npm run build
```

Expected: both commands pass.

- [ ] **Step 7: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/web
git commit -m "feat(final): add nextjs legal triage demo"
```

---

## Task 9: Write Final Report Markdown

**Files:**
- Create: `final_project/report/report.md`
- Modify: `final_project/report/results.md`
- Create: `final_project/report/references.md`

- [ ] **Step 1: Create references file**

Create `final_project/report/references.md`:

```markdown
# References

## Datasets

- Huang Liang Hsun. `lianghsun/tw-legal-synthetic-qa`. Hugging Face. https://huggingface.co/datasets/lianghsun/tw-legal-synthetic-qa
- Huang Liang Hsun. `lianghsun/tw-processed-law-article`. Hugging Face. https://huggingface.co/datasets/lianghsun/tw-processed-law-article
- Huang Liang Hsun. `lianghsun/tw-processed-law-ctx`. Hugging Face. https://huggingface.co/datasets/lianghsun/tw-processed-law-ctx
- Huang Liang Hsun. `lianghsun/tw-legal-nlp`. Hugging Face. https://huggingface.co/datasets/lianghsun/tw-legal-nlp
- Huang Liang Hsun. `lianghsun/tw-legal-benchmark-v1`. Hugging Face. https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1

## Models and Libraries

- scikit-learn. https://scikit-learn.org/
- Hugging Face Transformers. https://huggingface.co/docs/transformers/
- `hfl/chinese-macbert-base`. https://huggingface.co/hfl/chinese-macbert-base

## Related Products

- Lawbot AI. https://lawbot.tw/landing
- Lawsnote. https://about.lawsnote.com/
- LawChat. https://lawchat.com.tw/
- EasyLaw. https://easy-law.net/
- LawAI 法詢. https://www.lawfavor.com/
```

- [ ] **Step 2: Create report markdown**

Create `final_project/report/report.md`:

```markdown
# Chinese Legal Issue Triage: Multi-label Classification from Everyday Legal Narratives

## 1. Research Question

Many people describe legal problems in everyday Chinese but do not know which legal issue category their situation belongs to. This project asks:

> Can an NLP classifier map a Chinese legal scenario to one or more legal issue labels before any legal advice or retrieval step happens?

The goal is not to build a legal chatbot. The goal is to build an intake triage classifier that helps route a scenario to likely legal issue labels such as DUI, fraud, injury, divorce, tort damages, or contract breach.

## 2. Practical Application

Taiwan legal AI products often focus on legal question answering, judgment search, contract review, or document generation. These systems still need an earlier step: understanding what kind of issue the user is describing.

Potential users include:

- legal information websites that need issue routing,
- legal aid intake services,
- law firms collecting initial client facts,
- students learning to connect facts with legal concepts.

The classifier can suggest issue labels and related law articles as lookup clues. It does not provide legal advice.

## 3. Dataset Description

The main dataset is `lianghsun/tw-legal-synthetic-qa`, which contains 9,631 Chinese legal scenario and analysis pairs in ShareGPT-style message format. The user message is treated as the scenario. The assistant answer is used only as a weak-label source because it often contains legal citations.

The auxiliary dataset is `lianghsun/tw-processed-law-article`, which contains 230,974 Taiwan statute rows with fields such as `text`, `name`, `level`, `abandon_note`, and `modified_date`. This dataset is used for law article lookup, citation normalization, and ontology construction, not as a RAG corpus.

Limitations:

- Labels are weak labels extracted from synthetic legal analysis, not manually annotated gold labels.
- The data is biased toward criminal and civil issues.
- Procedural law labels from criminal or civil procedure are excluded from the first version because the project focuses on everyday issue triage, not court procedure classification.

## 4. NLP Pipeline

The pipeline has five main stages:

1. Citation normalization: normalize Chinese citation formats such as `刑法第185條之3` into stable law/article keys.
2. Ontology construction: map high-frequency law articles to 15-25 issue labels.
3. Weak labeling: extract citations from assistant answers and map them to issue labels.
4. Classification: compare a rule-based baseline with TF-IDF + SVM.
5. Evaluation and analysis: report micro-F1, macro-F1, hamming loss, and error cases.

## 5. Methods

The rule-based baseline uses aliases and keywords from the ontology. It is interpretable but brittle when the user uses unexpected wording.

The TF-IDF + SVM model uses character n-gram features and a One-vs-Rest multi-label classifier. This method is stronger than keyword matching because it can learn recurring Chinese text patterns from the training data.

If time allows, a Chinese BERT model such as `hfl/chinese-macbert-base` can be fine-tuned with a sigmoid multi-label classification head.

## 6. Results

Insert `report/results.md` here after running experiments.

The main metrics are:

- Micro-F1: overall performance dominated by frequent labels.
- Macro-F1: performance across labels, important for class imbalance.
- Hamming loss: multi-label error rate.

## 7. Demo

The interactive demo uses a Flask API and a Next.js frontend with shadcn/ui. The user enters a Chinese legal scenario, and the interface shows:

- predicted legal issue labels,
- confidence scores,
- supporting law articles as lookup clues,
- matched keywords or explanation features,
- a clear disclaimer that the tool is not legal advice.

## 8. Discussion

The most useful output is not a generated legal answer but a structured issue triage result. This makes the task measurable and avoids competing directly with full legal AI products that rely on large proprietary legal databases and RAG systems.

Expected strengths:

- clear NLP classification task,
- real Chinese legal text,
- reproducible pipeline,
- practical intake-routing use case.

Expected weaknesses:

- weak-label noise,
- class imbalance,
- limited label set,
- no guarantee that the suggested law articles apply to the specific case.

## 9. Conclusion

This project demonstrates a Chinese NLP pipeline for legal issue triage. It transforms everyday legal narratives into multi-label issue predictions using citation-derived weak labels and classification models. The system is positioned as an intake and routing aid, not as a legal advice generator.

## References

See `references.md`.
```

- [ ] **Step 3: Verify report covers guideline**

Check that `report.md` includes:

```text
research question
NLP method
experiment and testing process
results and solution
discussion and conclusion
dataset source, preprocessing, limitations
practical application
references
```

Expected: every item appears in a section heading or paragraph.

- [ ] **Step 4: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/report/report.md final_project/report/references.md
git commit -m "docs(final): draft final report structure"
```

---

## Task 10: Prepare English Slides and Speaker Notes

**Files:**
- Create: `final_project/report/slides.md`

- [ ] **Step 1: Create slide outline**

Create `final_project/report/slides.md`:

```markdown
# Chinese Legal Issue Triage

## Slide 1: Title

Chinese Legal Issue Triage: Multi-label Classification from Everyday Legal Narratives

Speaker notes:
Hello everyone. This project builds a Chinese NLP classifier for legal issue triage. The goal is not legal advice generation. The goal is to understand a user's Chinese legal scenario and predict likely legal issue labels.

## Slide 2: Research Question

Can we classify everyday Chinese legal narratives into useful legal issue labels?

Speaker notes:
Many people do not know whether their problem is fraud, tort damages, contract breach, divorce, or another issue. Before retrieval or legal consultation, the system needs to understand the issue type.

## Slide 3: Practical Motivation

- Legal AI products often focus on question answering or judgment search.
- Intake triage is the earlier step.
- Useful for legal aid, law firm intake, and legal information platforms.

Speaker notes:
This positioning avoids building a weak chatbot. Instead, the classifier can be used before a chatbot, before lawyer matching, or before legal document search.

## Slide 4: Datasets

- Main: `tw-legal-synthetic-qa`
- Auxiliary: `tw-processed-law-article`
- Labels: weak labels from extracted citations

Speaker notes:
The user message is the scenario. The assistant analysis provides citations. We extract citations and map them to issue labels through an ontology.

## Slide 5: NLP Pipeline

1. Citation normalization
2. Ontology mapping
3. Weak labeling
4. Classification
5. Evaluation
6. Web demo

Speaker notes:
The pipeline is fully reproducible. The important design decision is to classify issue labels instead of exact law articles, because exact article labels are too sparse.

## Slide 6: Methods

- Rule-based baseline
- TF-IDF + SVM
- Optional Chinese BERT

Speaker notes:
The rule-based baseline is interpretable but limited. TF-IDF and SVM can learn broader text patterns. BERT is optional if time and compute allow.

## Slide 7: Results

Insert metrics table from `report/results.md`.

Speaker notes:
I compare micro-F1, macro-F1, precision, recall, and hamming loss. Macro-F1 is important because the label distribution is imbalanced.

## Slide 8: Demo

Show Next.js interface:

- input Chinese scenario
- predicted issue labels
- confidence
- supporting law articles
- not legal advice disclaimer

Speaker notes:
The demo visualizes the classifier output. It does not generate legal advice. Supporting law articles are lookup clues only.

## Slide 9: Error Analysis

- synonym mismatch
- ambiguous scenario
- weak-label noise
- rare labels

Speaker notes:
The model can fail when the user wording is different from the training data, or when one scenario contains multiple legal issues. Weak labels also introduce noise.

## Slide 10: Conclusion

- Built a Chinese legal issue triage classifier
- Compared NLP methods
- Created an interactive demo
- Future work: stronger annotation, BERT, OOD tests

Speaker notes:
This project shows a focused and measurable NLP task. It is practical because issue triage is needed before legal search or legal consultation.
```

- [ ] **Step 2: Time-check speaker notes**

Read notes aloud once.

Expected: 8-10 minutes. If under 8 minutes, add one example and one error case. If over 10 minutes, shorten competitor and dataset details.

- [ ] **Step 3: Commit**

```bash
cd /home/roy422/NLP_homework
git add final_project/report/slides.md
git commit -m "docs(final): draft presentation slides and speaker notes"
```

---

## Task 11: Final Verification

**Files:**
- Modify as needed based on failures.

- [ ] **Step 1: Run Python tests**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run model pipeline**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python scripts/build_labels.py
uv run python scripts/split_data.py
uv run python baselines/rule.py
uv run python baselines/tfidf_svm.py
uv run python eval/metrics.py
```

Expected: no exceptions; `report/results.md` updates.

- [ ] **Step 3: Run Flask API**

Run:

```bash
cd /home/roy422/NLP_homework/final_project
uv run python api/app.py
```

Expected: Flask starts on `http://127.0.0.1:5000`.

- [ ] **Step 4: Run Next.js frontend**

In a second terminal:

```bash
cd /home/roy422/NLP_homework/final_project/web
npm run dev
```

Expected: frontend starts on `http://localhost:3000`.

- [ ] **Step 5: Manual demo smoke test**

Open `http://localhost:3000` and test:

```text
我酒後開車被警察攔下，酒測超標，想知道可能涉及什麼法律問題。
```

Expected: prediction includes `酒駕 / 公共危險`.

- [ ] **Step 6: Verify guideline coverage**

Check final deliverables:

```text
Code implementation: final_project scripts, API, frontend
Presentation slides: final_project/report/slides.md
Oral presentation: 8-10 minute English notes in slides.md
Dataset description: final_project/report/report.md section 3
NLP methods: final_project/report/report.md section 5
Experiment/testing process: final_project/report/report.md section 4 and results.md
Results and solution: final_project/report/report.md section 6
Discussion/conclusion: final_project/report/report.md sections 8-9
```

Expected: every guideline requirement is covered.

- [ ] **Step 7: Commit final docs/results**

```bash
cd /home/roy422/NLP_homework
git add final_project docs/superpowers/plans/2026-06-06-final-project-delivery.md
git commit -m "docs(final): add final delivery implementation plan"
```

---

## Self-Review

**Spec coverage:** This plan covers code implementation, Chinese NLP task, dataset description, preprocessing, classification methods, evaluation, practical application, interactive demo, report, slides, and 8-10 minute English speaker notes.

**Placeholder scan:** No task contains `TBD`, `TODO`, `implement later`, or "write tests" without code. Optional BERT is explicitly marked optional and excluded from MUST HAVE.

**Type consistency:** API response types in Flask and TypeScript use the same fields: `input`, `status`, `final_prediction`, `models`, `explanation`, and `disclaimer`.

