# 中文法律議題多標籤分類 — Implementation Plan

> **For agentic workers:** 此 plan 可用 `superpowers:executing-plans` skill 逐步執行，或直接人工按步驟做。每個 Task 下的 step 用 checkbox（`- [ ]`）追蹤。

**Goal**：實作一個輸入中文法律情境敘述、輸出相關法律議題（multi-label）的分類系統，比較 rule-based / TF-IDF+SVM / BERT 三法在不平衡資料下的表現。

**Architecture**：資料驅動建構議題 ontology → 從 `tw-legal-synthetic-qa` 用 regex 抽法條引用 → 映射議題 label → 三法訓練 → 跨文風 OOD 測試（aigrant）+ 錯誤分析 + Demo。

**Tech Stack**：Python 3.12 / uv / jieba / scikit-learn / HuggingFace transformers / Chinese MacBERT / Jupyter

**相關文件**：
- Spec：`FINAL_PROJECT_SPEC.md`
- 資料驗證：`final_project/DATA_USABILITY.md`

---

## Phase 0：專案初始化

### Task 0：Bootstrap `final_project/` 骨架

**Files:**
- Create: `final_project/README.md`
- Create: `final_project/pyproject.toml`
- Create: `final_project/.gitignore`
- Create: `final_project/data/` `final_project/scripts/` `final_project/baselines/` `final_project/models/` `final_project/eval/` `final_project/report/` `final_project/tests/`

- [ ] **Step 1：Create directory structure**

```bash
cd /Users/lubaiyu/Desktop/nlp_homework/final_project
mkdir -p data scripts baselines models eval report tests data/raw
touch data/.gitkeep scripts/.gitkeep baselines/.gitkeep models/.gitkeep eval/.gitkeep tests/.gitkeep
```

- [ ] **Step 2：Write pyproject.toml**

```toml
[project]
name = "legal-issue-classifier"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "datasets>=4.0",
    "jieba>=0.42",
    "scikit-learn>=1.5",
    "numpy>=1.26",
    "pandas>=2.0",
    "transformers>=4.40",
    "torch>=2.2",
    "accelerate>=0.30",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "tqdm>=4.66",
    "pytest>=8.0",
]
```

- [ ] **Step 3：Write README (專案說明 + 資料流程圖)**

內容至少包含：題目、使用者故事、資料來源（含授權）、如何重現（install + 各 script run command）、作者。

- [ ] **Step 4：Update root .gitignore**

```
# append to existing .gitignore
final_project/data/raw/
final_project/data/hf_cache/
final_project/models/**/checkpoint-*/
final_project/models/**/*.pt
*.DS_Store
.hf_token
```

- [ ] **Step 5：Install dependencies**

```bash
cd /Users/lubaiyu/Desktop/nlp_homework
source .venv/bin/activate
cd final_project && uv pip install -e .
```
Expected: all packages install, no errors.

- [ ] **Step 6：Commit**

```bash
git add final_project/
git commit -m "feat(final): bootstrap project skeleton and dependencies"
```

---

## Phase 1：資料準備（Step 1–3）

### Task 1：Regex 法條引用抽取器（含 unit test）

**Files:**
- Create: `final_project/scripts/citation_extractor.py`
- Create: `final_project/tests/test_citation_extractor.py`

- [ ] **Step 1：Write failing test (`tests/test_citation_extractor.py`)**

```python
import pytest
from scripts.citation_extractor import extract_citations

def test_extract_single_arabic():
    text = "根據刑法第185條之3的規定，酒駕屬公共危險罪。"
    assert extract_citations(text) == {("刑法", 185)}

def test_extract_chinese_numeral():
    text = "依民法第一百八十四條，侵權行為應負損害賠償責任。"
    assert extract_citations(text) == {("民法", 184)}

def test_extract_multiple():
    text = "刑法第310條誹謗罪，同法第313條妨害信用，民法第195條。"
    assert extract_citations(text) == {("刑法", 310), ("刑法", 313), ("民法", 195)}

def test_normalize_law_name():
    text = "中華民國刑法第185條"
    assert extract_citations(text) == {("刑法", 185)}

def test_no_match():
    text = "今天天氣真好，去爬山好不好？"
    assert extract_citations(text) == set()

def test_dedup():
    text = "刑法第185條很重要。又根據刑法第185條……"
    assert extract_citations(text) == {("刑法", 185)}
```

- [ ] **Step 2：Run test — verify fail**

```bash
cd final_project && pytest tests/test_citation_extractor.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3：Implement `scripts/citation_extractor.py`**

```python
"""Extract (law_name, article_number) pairs from Chinese legal text."""
import re
from typing import Set, Tuple

LAW_NAMES = [
    "中華民國刑法", "刑法", "民法", "勞動基準法", "勞基法",
    "性別工作平等法", "性別平等工作法", "就業服務法",
    "勞工保險條例", "勞工退休金條例", "勞資爭議處理法",
    "職業安全衛生法", "就業保險法", "民事訴訟法", "刑事訴訟法",
    "行政訴訟法", "行政程序法", "家事事件法", "家庭暴力防治法",
    "道路交通管理處罰條例", "毒品危害防制條例", "公司法",
    "專利法", "商標法", "著作權法", "消費者保護法",
    "個人資料保護法", "公平交易法", "證券交易法", "票據法",
    "保險法", "所得稅法", "兒童及少年福利與權益保障法",
]
# longer first to avoid prefix collisions
LAW_NAMES_SORTED = sorted(LAW_NAMES, key=len, reverse=True)

_PATTERN = re.compile(
    r"(" + "|".join(LAW_NAMES_SORTED) + r")"
    r"\s*第\s*([0-9零一二三四五六七八九十百千]+)\s*條"
)

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

_NORMALIZE = {
    "中華民國刑法": "刑法",
    "勞基法": "勞動基準法",
    "性別平等工作法": "性別工作平等法",
}


def _cn_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    n = 0
    cur = 0
    for c in s:
        if c in _CN_DIGIT:
            cur = _CN_DIGIT[c]
        elif c == "十":
            n += (cur or 1) * 10
            cur = 0
        elif c == "百":
            n += (cur or 1) * 100
            cur = 0
        elif c == "千":
            n += (cur or 1) * 1000
            cur = 0
    return n + cur


def extract_citations(text: str) -> Set[Tuple[str, int]]:
    """Return deduped set of (law_name, article_number) pairs."""
    out = set()
    for law, num in _PATTERN.findall(text):
        law = _NORMALIZE.get(law, law)
        out.add((law, _cn_to_int(num)))
    return out
```

- [ ] **Step 4：Run test — verify pass**

```bash
cd final_project && pytest tests/test_citation_extractor.py -v
```
Expected: all 6 tests pass.

- [ ] **Step 5：Commit**

```bash
git add final_project/scripts/citation_extractor.py final_project/tests/
git commit -m "feat(final): add citation extractor with normalization + tests"
```

---

### Task 2：從主資料集抽取 pair 頻率統計

**Files:**
- Create: `final_project/scripts/analyze_pair_freq.py`
- Create: `final_project/data/pair_freq.json`

- [ ] **Step 1：Write `scripts/analyze_pair_freq.py`**

```python
"""Load tw-legal-synthetic-qa, extract all citations, save frequency table."""
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from scripts.citation_extractor import extract_citations

ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
rows = list(ds["train"]) + list(ds["test"])

pair_counter = Counter()
law_counter = Counter()
rows_with_label = 0
for r in rows:
    text = r["messages"][0]["content"] + "\n" + r["messages"][1]["content"]
    pairs = extract_citations(text)
    if pairs:
        rows_with_label += 1
    for law, num in pairs:
        pair_counter[f"{law}§{num}"] += 1
        law_counter[law] += 1

out = {
    "total_rows": len(rows),
    "rows_with_label": rows_with_label,
    "unique_pairs": len(pair_counter),
    "unique_laws": len(law_counter),
    "law_freq": dict(law_counter.most_common()),
    "pair_freq": dict(pair_counter.most_common()),
}
Path("data/pair_freq.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2)
)
print(f"Saved pair_freq.json  rows_with_label={rows_with_label}  unique_pairs={len(pair_counter)}")
```

- [ ] **Step 2：Run**

```bash
cd final_project && python -m scripts.analyze_pair_freq
```
Expected:
- `rows_with_label ≈ 3,158`
- `unique_pairs ≈ 813`
- `data/pair_freq.json` created

- [ ] **Step 3：Commit**

```bash
git add final_project/scripts/analyze_pair_freq.py final_project/data/pair_freq.json
git commit -m "feat(final): compute pair frequency statistics"
```

---

### Task 3：議題 Ontology 建構

**Files:**
- Create: `final_project/scripts/build_ontology_stub.py`（從 pair_freq 產出 ontology 草稿）
- Create: `final_project/data/issues.yaml`（**人工精修版**）

- [ ] **Step 1：Query 法條原文（輔助 ontology 設計）**

```python
# scripts/lookup_law_text.py
from datasets import load_dataset

def get_article_text(law_name: str, article_num: int) -> str:
    ds = load_dataset("lianghsun/tw-processed-law-article", split="train", streaming=True)
    for r in ds:
        if r.get("name", "").strip() == law_name and f"第 {article_num} 條" in r.get("text", ""):
            return r["text"]
    return ""

if __name__ == "__main__":
    import json, sys
    freq = json.loads(open("data/pair_freq.json").read())
    top = list(freq["pair_freq"].items())[:50]
    for pair, cnt in top:
        law, art = pair.split("§")
        txt = get_article_text(law, int(art))
        print(f"\n[{cnt}] {pair}\n{txt[:300]}")
```

Run: `python -m scripts.lookup_law_text > data/top50_laws.txt`
Expected: human-readable file listing top 50 pairs with their article text for ontology design.

- [ ] **Step 2：Write `scripts/build_ontology_stub.py`（自動初版）**

```python
"""Generate stub ontology from pair frequencies; human edits after."""
import json, yaml
from pathlib import Path
from collections import defaultdict

freq = json.loads(Path("data/pair_freq.json").read_text())
pair_freq = freq["pair_freq"]

# group by law prefix as initial clustering (human refines after)
groups = defaultdict(list)
for pair, cnt in pair_freq.items():
    law, art = pair.split("§")
    groups[law].append((int(art), cnt))

stub = []
for law, arts in groups.items():
    arts.sort(key=lambda x: -x[1])
    stub.append({
        "id": f"{law}_stub",
        "name": f"{law}（待人工細分）",
        "aliases": [],
        "laws": [[law, a] for a, _ in arts[:10]],
        "keywords": [],
        "_total_rows": sum(c for _, c in arts),
    })

Path("data/issues_stub.yaml").write_text(
    yaml.safe_dump(stub, allow_unicode=True, sort_keys=False)
)
print(f"Wrote stub with {len(stub)} law-level groups")
```

Run: `python -m scripts.build_ontology_stub`

- [ ] **Step 3：人工精修 `data/issues.yaml`**

以 `data/top50_laws.txt` + `data/issues_stub.yaml` 為基礎，產出 30–50 個議題。格式：

```yaml
- id: defamation
  name: 妨害名譽
  aliases: [誹謗, 公然侮辱, 說我壞話, 肉搜, 罵人]
  keywords: [名譽, 誹謗, 侮辱, 人格權]
  laws:
    - [刑法, 310]
    - [刑法, 311]
    - [刑法, 313]
    - [民法, 195]

- id: drunk_driving
  name: 酒駕/公共危險
  aliases: [酒駕, 酒測, 吹氣, 吐氣酒精濃度]
  keywords: [酒精, 駕駛, 公共危險, 吐氣]
  laws:
    - [刑法, 185]
    - [道路交通管理處罰條例, 35]
```

**預期產出約 30–50 條 issue**，涵蓋 ≥80% 的 pair_freq（前 150 pairs）。

- [ ] **Step 4：Write ontology validator**

```python
# tests/test_ontology.py
import yaml, pytest
from pathlib import Path

def test_ontology_loads():
    data = yaml.safe_load(Path("data/issues.yaml").read_text())
    assert len(data) >= 30, f"Expected ≥30 issues, got {len(data)}"
    assert len(data) <= 60, f"Too many issues: {len(data)}"

def test_all_ids_unique():
    data = yaml.safe_load(Path("data/issues.yaml").read_text())
    ids = [d["id"] for d in data]
    assert len(set(ids)) == len(ids)

def test_required_fields():
    data = yaml.safe_load(Path("data/issues.yaml").read_text())
    for item in data:
        assert "id" in item and "name" in item and "laws" in item
        assert len(item["laws"]) >= 1
        for law_pair in item["laws"]:
            assert len(law_pair) == 2
```

Run: `pytest tests/test_ontology.py -v`

- [ ] **Step 5：Commit**

```bash
git add final_project/scripts/lookup_law_text.py \
        final_project/scripts/build_ontology_stub.py \
        final_project/data/issues.yaml \
        final_project/data/issues_stub.yaml \
        final_project/data/top50_laws.txt \
        final_project/tests/test_ontology.py
git commit -m "feat(final): construct issue ontology (30-50 categories)"
```

---

### Task 4：自動打標籤器 + 測試

**Files:**
- Create: `final_project/scripts/auto_labeler.py`
- Create: `final_project/tests/test_auto_labeler.py`

- [ ] **Step 1：Write failing test**

```python
# tests/test_auto_labeler.py
from scripts.auto_labeler import map_citations_to_issues

FAKE_ONTOLOGY = [
    {"id": "defamation", "laws": [["刑法", 310], ["刑法", 313]]},
    {"id": "drunk_driving", "laws": [["刑法", 185]]},
]

def test_single_citation_maps_to_issue():
    citations = {("刑法", 310)}
    assert map_citations_to_issues(citations, FAKE_ONTOLOGY) == {"defamation"}

def test_multiple_citations_one_issue():
    citations = {("刑法", 310), ("刑法", 313)}
    assert map_citations_to_issues(citations, FAKE_ONTOLOGY) == {"defamation"}

def test_multi_issue():
    citations = {("刑法", 310), ("刑法", 185)}
    assert map_citations_to_issues(citations, FAKE_ONTOLOGY) == {"defamation", "drunk_driving"}

def test_unmatched_returns_empty():
    citations = {("刑法", 9999)}
    assert map_citations_to_issues(citations, FAKE_ONTOLOGY) == set()
```

- [ ] **Step 2：Run test — verify fail**

- [ ] **Step 3：Implement `scripts/auto_labeler.py`**

```python
"""Map (law, article) citations to issue IDs via ontology."""
import yaml
from pathlib import Path
from typing import Set, Tuple, List, Dict

def load_ontology(path: str = "data/issues.yaml") -> List[Dict]:
    return yaml.safe_load(Path(path).read_text())

def map_citations_to_issues(
    citations: Set[Tuple[str, int]],
    ontology: List[Dict],
) -> Set[str]:
    out = set()
    for issue in ontology:
        for law, art in issue["laws"]:
            if (law, int(art)) in citations:
                out.add(issue["id"])
                break
    return out
```

- [ ] **Step 4：Run tests — pass**

- [ ] **Step 5：Commit**

```bash
git add final_project/scripts/auto_labeler.py final_project/tests/test_auto_labeler.py
git commit -m "feat(final): map citations to issue labels"
```

---

### Task 5：產出 labeled.jsonl

**Files:**
- Create: `final_project/scripts/build_labeled_dataset.py`
- Create: `final_project/data/labeled.jsonl`

- [ ] **Step 1：Write `scripts/build_labeled_dataset.py`**

```python
"""Apply extractor + labeler to full dataset, write labeled.jsonl."""
import json
from pathlib import Path
from datasets import load_dataset
from scripts.citation_extractor import extract_citations
from scripts.auto_labeler import load_ontology, map_citations_to_issues

ontology = load_ontology()
ds = load_dataset("lianghsun/tw-legal-synthetic-qa")

written = 0
with open("data/labeled.jsonl", "w", encoding="utf-8") as f:
    for split in ["train", "test"]:
        for i, r in enumerate(ds[split]):
            user_msg = r["messages"][0]["content"]
            asst_msg = r["messages"][1]["content"]
            citations = extract_citations(user_msg + "\n" + asst_msg)
            labels = map_citations_to_issues(citations, ontology)
            if not labels:
                continue
            if len(user_msg) < 50:
                continue  # too short
            if "資訊不足" in asst_msg or "沒有提及" in asst_msg:
                continue  # skip non-answers
            f.write(json.dumps({
                "id": f"{split}-{i}",
                "story": user_msg,
                "labels": sorted(labels),
                "raw_citations": [[l, a] for l, a in sorted(citations)],
            }, ensure_ascii=False) + "\n")
            written += 1
print(f"Wrote {written} labeled rows to data/labeled.jsonl")
```

- [ ] **Step 2：Run**

```bash
cd final_project && python -m scripts.build_labeled_dataset
```
Expected: ~2,500–3,000 rows written (after filters).

- [ ] **Step 3：Manual audit — sample 50 rows**

```bash
shuf -n 50 data/labeled.jsonl > data/audit_sample.jsonl
```

人工讀一遍，目標 ≥ 42/50 (84%) label 對應合理。若低於此：回 Task 3 調整 ontology 後重跑。

- [ ] **Step 4：Commit**

```bash
git add final_project/scripts/build_labeled_dataset.py final_project/data/labeled.jsonl
git commit -m "feat(final): produce labeled training dataset"
```

---

## Phase 1.5：資料擴充（可選，依 Task 5 產出量決定）

> **觸發條件**：Task 5 完成後檢查 `labeled.jsonl`。若符合下列任一條件 → 執行對應擴充 task：
> - 總筆數 < 2,500 → 跑 Task 5a + 5b
> - 刑法類別佔比 > 70% → 跑 Task 5b（強制補非刑法樣本）
> - 任何議題 < 30 筆 → 跑 Task 5c
> - 任何議題 < 5 筆 → 跑 Task 5d
>
> **全部成本**：10 分鐘申請 + 半天合成，預期多 1,500–2,000 筆、類別平衡改善。

### Task 5a：申請 gated HuggingFace 資料集（10–15 min，可選）

**目標**：解鎖 `tw-legal-qa-3M` (5.4k), `tw-legal-qa-chat` (527) 等同分布資料集。

**Files:**
- Create: `.env`（記 HF_TOKEN，已在 .gitignore）
- Modify: `final_project/scripts/build_labeled_dataset.py`

- [ ] **Step 1：Apply for dataset access**

在瀏覽器去下列頁面點 "Request Access"（通常自動核准）：
- https://huggingface.co/datasets/lianghsun/tw-legal-qa-3M
- https://huggingface.co/datasets/lianghsun/tw-legal-qa-chat

- [ ] **Step 2：建立 HF token**

去 https://huggingface.co/settings/tokens 建 `Read` token，存入專案：

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxx" > .env
huggingface-cli login --token hf_xxxxxxxxxxxx
```

- [ ] **Step 3：驗證可載入**

```python
from datasets import load_dataset
ds = load_dataset("lianghsun/tw-legal-qa-3M")
print(ds)
```
若能印出 splits → 核准成功。

- [ ] **Step 4：擴充 `build_labeled_dataset.py`**

在 Task 5 的 script 底下加：

```python
# append gated datasets
for name in ["lianghsun/tw-legal-qa-3M", "lianghsun/tw-legal-qa-chat"]:
    try:
        ds = load_dataset(name)
    except Exception as e:
        print(f"Skip {name}: {e}")
        continue
    for split in ds.keys():
        for i, r in enumerate(ds[split]):
            # tw-legal-qa-3M: 'text' field combining user+answer
            # tw-legal-qa-chat: 'input' + 'output'
            if "text" in r:
                story = r["text"][:400]      # heuristic: first 400 chars = scenario
                full = r["text"]
            else:
                story = r.get("input", "")
                full = story + "\n" + r.get("output", "")
            citations = extract_citations(full)
            labels = map_citations_to_issues(citations, ontology)
            if not labels or len(story) < 50: continue
            f.write(json.dumps({
                "id": f"{name.split('/')[-1]}-{split}-{i}",
                "story": story,
                "labels": sorted(labels),
                "source": name,
            }, ensure_ascii=False) + "\n")
            written += 1
```

- [ ] **Step 5：Rerun + Commit**

```bash
cd final_project && python -m scripts.build_labeled_dataset
wc -l data/labeled.jsonl   # 應該從 ~2800 增加到 ~5000+
git add scripts/build_labeled_dataset.py data/labeled.jsonl
git commit -m "feat(final): extend labeled dataset with gated sources"
```

---

### Task 5b：aigrant 切入訓練（補非刑法樣本）

**目標**：把 aigrant 591 筆「與主資料共享 label 的」切 300 訓練 + 291 OOD 測試；aigrant 分布幾乎沒刑法 → 強力補平衡。

**Files:**
- Create: `final_project/scripts/add_aigrant_train.py`
- Modify: `final_project/data/labeled.jsonl`
- Create: `final_project/data/ood_test.jsonl`

- [ ] **Step 1：Implement `scripts/add_aigrant_train.py`**

```python
"""Split aigrant usable rows 300/291 → train + OOD test."""
import json, random
from pathlib import Path
from datasets import load_dataset
from scripts.auto_labeler import load_ontology

random.seed(42)
ontology = load_ontology()
# issue → set of law names that appear in its law list
issue_law_names = {i["id"]: {p[0] for p in i["laws"]} for i in ontology}

ds = load_dataset("aigrant/taiwan-ly-law-research")
usable = []
for r in ds["train"]:
    rl = r.get("related_laws", "") or ""
    laws = [x.strip() for x in rl.split(";") if x.strip()]
    issues = set()
    for iid, lawset in issue_law_names.items():
        if lawset & set(laws):
            issues.add(iid)
    if not issues: continue
    title = r.get("title", "") or ""
    content = r.get("content", "") or ""
    # Use title + first 300 chars of content → story-like input
    story = (title + "。" + content)[:400]
    if len(story) < 50: continue
    usable.append({
        "id": f"aigrant-{r['research_no']}",
        "story": story,
        "labels": sorted(issues),
        "source": "aigrant",
    })

random.shuffle(usable)
cut = int(len(usable) * 0.5)  # roughly half to train
train_part, ood_part = usable[:cut], usable[cut:]

# append train_part to labeled.jsonl
with open("data/labeled.jsonl", "a", encoding="utf-8") as f:
    for r in train_part:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
# write OOD separately (do NOT include in labeled.jsonl)
with open("data/ood_test.jsonl", "w", encoding="utf-8") as f:
    for r in ood_part:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Appended {len(train_part)} train rows from aigrant")
print(f"Wrote {len(ood_part)} to data/ood_test.jsonl")
```

- [ ] **Step 2：Run**

```bash
cd final_project && python -m scripts.add_aigrant_train
```
Expected: 約 300 筆加到 labeled，291 筆寫到 ood_test.jsonl。

- [ ] **Step 3：Modify `eval/run_all.py`**

把 OOD 部分改成讀 `data/ood_test.jsonl` 而不是動態抽（原 Task 11 寫的 aigrant 動態抽改掉）：

```python
# 取代原本 ood_rows 動態抽的那段
ood_rows = [json.loads(l) for l in Path("data/ood_test.jsonl").read_text().splitlines()]
X_ood = [r["story"] for r in ood_rows]
y_ood = [set(r["labels"]) for r in ood_rows]
```

- [ ] **Step 4：Commit**

```bash
git add scripts/add_aigrant_train.py data/labeled.jsonl data/ood_test.jsonl eval/run_all.py
git commit -m "feat(final): add aigrant training subset + fixed OOD test set"
```

---

### Task 5c：LLM 合成稀有類別（補到每類 ≥ 50 筆）

**目標**：對樣本 < 30 筆的議題生成變體，補平類別。

**前置**：先跑一次 Task 6 切 train，看 per-class count。

**Files:**
- Create: `final_project/scripts/llm_augment.py`
- Create: `final_project/data/prompts/augment.md`
- Modify: `final_project/data/labeled.jsonl`

- [ ] **Step 1：統計哪些類別缺**

```python
# scripts/find_rare_classes.py
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("data/labeled.jsonl").read_text().splitlines()]
cnt = Counter(l for r in rows for l in r["labels"])
rare = [(iid, c) for iid, c in cnt.items() if c < 30]
print(f"Rare classes (<30 rows): {len(rare)}")
for iid, c in sorted(rare, key=lambda x: x[1]):
    print(f"  {iid}: {c}")
```

- [ ] **Step 2：Write prompt template `data/prompts/augment.md`**

```
你是法律諮詢網站的編輯。我會給你一個議題代碼、該議題的描述、3 個真實範例情境。
請你模仿這 3 個範例的語氣、長度、主角命名慣例，生成 5 個**完全不同故事**但仍指向同一議題的情境。

規則：
1. 每個故事 100–300 字
2. 提及相關法律關鍵字（可略，不是每個都要）
3. 口語化，像一般人在諮詢
4. 不要重複原範例的場景
5. 不要輸出任何解釋、只輸出 5 個故事用 ---SPLIT--- 分隔

議題：{issue_id}
描述：{issue_name}
關鍵字：{keywords}
範例 1：{example_1}
範例 2：{example_2}
範例 3：{example_3}
```

- [ ] **Step 3：Implement `scripts/llm_augment.py`**

```python
"""Use LLM to generate synthetic samples for rare classes."""
import json, os
from pathlib import Path
from openai import OpenAI  # OpenAI-compatible SDK for Cerebras

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ["CEREBRAS_API_KEY"],
)
prompt_tpl = Path("data/prompts/augment.md").read_text()
rare_threshold = 30

rows = [json.loads(l) for l in Path("data/labeled.jsonl").read_text().splitlines()]
from collections import defaultdict, Counter
by_issue = defaultdict(list)
for r in rows:
    for l in r["labels"]:
        by_issue[l].append(r)
cnt = Counter({k: len(v) for k, v in by_issue.items()})

with open("data/labeled.jsonl", "a", encoding="utf-8") as out:
    for iid, samples in by_issue.items():
        if cnt[iid] >= rare_threshold: continue
        examples = samples[:3]
        prompt = prompt_tpl.format(
            issue_id=iid,
            issue_name=iid,  # TODO: lookup from ontology name
            keywords="",
            example_1=examples[0]["story"][:300] if len(examples) > 0 else "",
            example_2=examples[1]["story"][:300] if len(examples) > 1 else "",
            example_3=examples[2]["story"][:300] if len(examples) > 2 else "",
        )
        resp = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8, max_tokens=2000,
        )
        text = resp.choices[0].message.content
        stories = [s.strip() for s in text.split("---SPLIT---") if len(s.strip()) > 50]
        for i, s in enumerate(stories):
            out.write(json.dumps({
                "id": f"synth-{iid}-{i}",
                "story": s,
                "labels": [iid],
                "source": "llm_synthetic",
            }, ensure_ascii=False) + "\n")
        print(f"{iid}: {cnt[iid]} → +{len(stories)}")
```

- [ ] **Step 4：抽樣人工驗證**

```bash
# 抽 30 筆合成資料檢查
grep '"source": "llm_synthetic"' data/labeled.jsonl | shuf -n 30 > data/synth_audit.jsonl
```
目標 ≥ 25/30 合理；若差，降 temperature 重跑或調 prompt。

- [ ] **Step 5：Commit**

```bash
git add scripts/llm_augment.py data/prompts/augment.md data/labeled.jsonl
git commit -m "feat(final): LLM synthetic augmentation for rare classes"
```

---

### Task 5d：合併極稀有類別（最後手段）

**觸發條件**：上述擴充後仍有議題 < 5 筆 → 合併成「其他X」。

**Files:**
- Modify: `final_project/data/issues.yaml`
- Modify: `final_project/data/labeled.jsonl`
- Create: `final_project/scripts/merge_rare_issues.py`

- [ ] **Step 1：Implement merger**

```python
# scripts/merge_rare_issues.py
import json, yaml
from collections import Counter, defaultdict
from pathlib import Path

# 定義合併規則（手動，基於 ontology 的領域分類）
MERGE_MAP = {
    # rare issue → super-bucket
    "bounced_check": "other_commercial",
    "property_division": "other_civil",
    # 依實際 Task 5c 結果補
}

rows = [json.loads(l) for l in Path("data/labeled.jsonl").read_text().splitlines()]
for r in rows:
    r["labels"] = sorted({MERGE_MAP.get(l, l) for l in r["labels"]})

with open("data/labeled.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 同步更新 issues.yaml（刪除被合併的、新增 super-bucket）
ont = yaml.safe_load(Path("data/issues.yaml").read_text())
kept = [i for i in ont if i["id"] not in MERGE_MAP]
for super_id in set(MERGE_MAP.values()):
    if not any(i["id"] == super_id for i in kept):
        kept.append({"id": super_id, "name": super_id, "aliases": [], "keywords": [], "laws": []})
Path("data/issues.yaml").write_text(yaml.safe_dump(kept, allow_unicode=True, sort_keys=False))
```

- [ ] **Step 2：Run + Verify**

```bash
python -m scripts.merge_rare_issues
python -m scripts.find_rare_classes  # 重新統計，應無 <5 的
```

- [ ] **Step 3：Commit**

```bash
git add scripts/merge_rare_issues.py data/issues.yaml data/labeled.jsonl
git commit -m "feat(final): merge ultra-rare classes into bucket categories"
```

---

### Task 6：資料切分（stratified）

**Files:**
- Create: `final_project/scripts/split_data.py`
- Create: `final_project/tests/test_split.py`
- Create: `final_project/data/{train,val,test}.jsonl`
- Create: `final_project/data/stats.md`

- [ ] **Step 1：Write test**

```python
# tests/test_split.py
import json
from pathlib import Path

def test_no_id_leakage():
    splits = {}
    for name in ["train", "val", "test"]:
        lines = Path(f"data/{name}.jsonl").read_text().splitlines()
        splits[name] = {json.loads(l)["id"] for l in lines}
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])

def test_split_sizes_reasonable():
    lines = Path("data/labeled.jsonl").read_text().splitlines()
    total = len(lines)
    train = len(Path("data/train.jsonl").read_text().splitlines())
    val = len(Path("data/val.jsonl").read_text().splitlines())
    test = len(Path("data/test.jsonl").read_text().splitlines())
    assert abs(train/total - 0.70) < 0.05
    assert abs(val/total - 0.15) < 0.05
    assert abs(test/total - 0.15) < 0.05
```

- [ ] **Step 2：Implement `scripts/split_data.py` (multi-label stratified)**

Use `iterative-stratification` library OR fall back to `sklearn.model_selection.train_test_split` with multi-label → random stratify hack.

```python
"""70/15/15 stratified split for multi-label dataset."""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

rows = [json.loads(l) for l in Path("data/labeled.jsonl").read_text().splitlines()]
# stratification proxy: use first label
strata = [r["labels"][0] for r in rows]

# drop classes with <3 samples so stratify works
cnt = Counter(strata)
keep = [i for i, s in enumerate(strata) if cnt[s] >= 3]
rows = [rows[i] for i in keep]
strata = [strata[i] for i in keep]

train, rest, s_train, s_rest = train_test_split(
    rows, strata, test_size=0.30, random_state=42, stratify=strata)
val, test = train_test_split(
    rest, test_size=0.50, random_state=42, stratify=s_rest)

for name, data in [("train", train), ("val", val), ("test", test)]:
    with open(f"data/{name}.jsonl", "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} rows")
```

Run: `python -m scripts.split_data`

- [ ] **Step 3：Run test — verify pass**

```bash
pytest tests/test_split.py -v
```

- [ ] **Step 4：Generate `data/stats.md`**

```python
# scripts/write_stats.py
import json
from collections import Counter
from pathlib import Path

out = ["# Dataset Statistics\n"]
for split in ["train", "val", "test"]:
    rows = [json.loads(l) for l in Path(f"data/{split}.jsonl").read_text().splitlines()]
    labels = Counter(l for r in rows for l in r["labels"])
    out.append(f"\n## {split} ({len(rows)} rows)\n")
    for lab, cnt in labels.most_common():
        out.append(f"- {lab}: {cnt}")
Path("data/stats.md").write_text("\n".join(out))
```

Run: `python -m scripts.write_stats`

- [ ] **Step 5：Commit**

```bash
git add final_project/scripts/split_data.py \
        final_project/scripts/write_stats.py \
        final_project/tests/test_split.py \
        final_project/data/{train,val,test}.jsonl \
        final_project/data/stats.md
git commit -m "feat(final): stratified 70/15/15 split + stats"
```

---

## Phase 2：Baselines (Step 4–5)

### Task 7：Rule-based Baseline

**Files:**
- Create: `final_project/baselines/rule.py`
- Create: `final_project/tests/test_rule_baseline.py`

- [ ] **Step 1：Write test**

```python
# tests/test_rule_baseline.py
from baselines.rule import RuleClassifier

ONTOLOGY = [
    {"id": "defamation", "aliases": ["誹謗", "名譽"], "keywords": ["名譽"]},
    {"id": "drunk", "aliases": ["酒駕", "酒測"], "keywords": ["酒精"]},
]

def test_single_hit():
    clf = RuleClassifier(ONTOLOGY)
    assert clf.predict("他誹謗我") == {"defamation"}

def test_multi_hit():
    clf = RuleClassifier(ONTOLOGY)
    assert clf.predict("他又誹謗又酒駕") == {"defamation", "drunk"}

def test_no_hit():
    clf = RuleClassifier(ONTOLOGY)
    assert clf.predict("天氣很好") == set()
```

- [ ] **Step 2：Implement `baselines/rule.py`**

```python
"""Rule-based baseline: match aliases + keywords."""
from typing import List, Dict, Set

class RuleClassifier:
    def __init__(self, ontology: List[Dict]):
        self.ontology = ontology
    def predict(self, text: str) -> Set[str]:
        out = set()
        for issue in self.ontology:
            for kw in issue.get("aliases", []) + issue.get("keywords", []):
                if kw in text:
                    out.add(issue["id"])
                    break
        return out
    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]
```

- [ ] **Step 3：Smoke test on 10 samples**

```bash
cd final_project && python -c "
import json
from scripts.auto_labeler import load_ontology
from baselines.rule import RuleClassifier
clf = RuleClassifier(load_ontology())
for line in open('data/test.jsonl').readlines()[:10]:
    r = json.loads(line)
    print(f'{r[\"labels\"]} | predicted: {clf.predict(r[\"story\"])}')"
```
Expected: 能看到預測輸出（好壞是 Task 11 的事）。

> **備註**：正式評估在 Task 11（`eval/run_all.py`）一起跑三法比較。
> 這裡只確認 classifier 類別能用、不報錯。

- [ ] **Step 4：Commit**

```bash
git add final_project/baselines/rule.py \
        final_project/tests/test_rule_baseline.py
git commit -m "feat(final): rule-based baseline + test"
```

---

### Task 8：TF-IDF + Linear Classifier

**Files:**
- Create: `final_project/baselines/tfidf_svm.py`
- Create: `final_project/baselines/custom_jieba_dict.txt`
- Create: `final_project/models/tfidf_svm.pkl`

- [ ] **Step 1：Write jieba custom dictionary**

```
# custom_jieba_dict.txt — 法律專有名詞
勞基法 100 n
勞動基準法 100 n
妨害名譽 100 n
妨害信用 100 n
資遣費 100 n
公共危險 100 n
酒駕 100 n
肇事逃逸 100 n
家暴 100 n
性騷擾 100 n
告訴乃論 100 n
連帶保證 100 n
...
```

- [ ] **Step 2：Implement `baselines/tfidf_svm.py`**

```python
"""TF-IDF + OvR LogisticRegression / LinearSVC."""
import json, pickle, argparse
from pathlib import Path
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

jieba.load_userdict("baselines/custom_jieba_dict.txt")

def tokenize(text: str) -> str:
    return " ".join(jieba.cut(text))

def load_split(name):
    rows = [json.loads(l) for l in Path(f"data/{name}.jsonl").read_text().splitlines()]
    return [r["story"] for r in rows], [r["labels"] for r in rows]

def train(model_kind="lr"):
    X_tr, y_tr = load_split("train")
    X_val, y_val = load_split("val")
    X_tr = [tokenize(t) for t in X_tr]
    X_val = [tokenize(t) for t in X_val]

    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=30000)
    Xv_tr = vec.fit_transform(X_tr)
    Xv_val = vec.transform(X_val)

    mlb = MultiLabelBinarizer()
    Yt = mlb.fit_transform(y_tr)
    Yv = mlb.transform(y_val)

    base = LogisticRegression(max_iter=1000, C=1.0) if model_kind == "lr" \
           else LinearSVC(C=1.0)
    clf = OneVsRestClassifier(base)
    clf.fit(Xv_tr, Yt)

    with open(f"models/tfidf_{model_kind}.pkl", "wb") as f:
        pickle.dump({"vec": vec, "clf": clf, "mlb": mlb}, f)
    print(f"Saved models/tfidf_{model_kind}.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "svm"], default="lr")
    args = ap.parse_args()
    train(args.model)
```

- [ ] **Step 3：Train both (LR + SVM)**

```bash
cd final_project
python -m baselines.tfidf_svm --model lr
python -m baselines.tfidf_svm --model svm
```

- [ ] **Step 4：Threshold tuning script**

For LR (probabilistic), sweep threshold ∈ {0.2, 0.3, 0.4, 0.5} on val set, pick best macro-F1.

- [ ] **Step 5：Commit**

```bash
git add final_project/baselines/tfidf_svm.py \
        final_project/baselines/custom_jieba_dict.txt \
        final_project/models/tfidf_lr.pkl \
        final_project/models/tfidf_svm.pkl
git commit -m "feat(final): TF-IDF + LR/SVM baselines with threshold tuning"
```

---

## Phase 3：BERT Fine-tune (Step 6)

### Task 9：BERT Multi-label Fine-tune

**Files:**
- Create: `final_project/models/bert_finetune.py`
- Create: `final_project/models/bert_config.yaml`

- [ ] **Step 1：Write config**

```yaml
# models/bert_config.yaml
model_name: hfl/chinese-macbert-base
max_length: 512
batch_size: 16
eval_batch_size: 32
learning_rate: 2e-5
num_epochs: 5
weight_decay: 0.01
warmup_ratio: 0.1
early_stopping_patience: 2
output_dir: models/bert_ckpt
```

- [ ] **Step 2：Implement `models/bert_finetune.py`**

```python
"""BERT multi-label fine-tune with HuggingFace Trainer."""
import json, yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss

cfg = yaml.safe_load(Path("models/bert_config.yaml").read_text())

def load_split(name):
    rows = [json.loads(l) for l in Path(f"data/{name}.jsonl").read_text().splitlines()]
    return [r["story"] for r in rows], [r["labels"] for r in rows]

X_tr, y_tr = load_split("train")
X_val, y_val = load_split("val")

mlb = MultiLabelBinarizer()
Yt = mlb.fit_transform(y_tr).astype("float32")
Yv = mlb.transform(y_val).astype("float32")
n_labels = len(mlb.classes_)

tok = AutoTokenizer.from_pretrained(cfg["model_name"])

class MLDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, padding=False,
                       max_length=cfg["max_length"])
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i].tolist()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.from_numpy(logits)).numpy() > 0.5).astype(int)
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "hamming": hamming_loss(labels, preds),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    cfg["model_name"], num_labels=n_labels, problem_type="multi_label_classification")

args = TrainingArguments(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["num_epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    per_device_eval_batch_size=cfg["eval_batch_size"],
    learning_rate=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],
    warmup_ratio=cfg["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=MLDataset(X_tr, Yt), eval_dataset=MLDataset(X_val, Yv),
    tokenizer=tok, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(cfg["early_stopping_patience"])],
)
trainer.train()
trainer.save_model(cfg["output_dir"] + "/best")
Path(cfg["output_dir"] + "/label_classes.json").write_text(
    json.dumps(list(mlb.classes_), ensure_ascii=False))
```

- [ ] **Step 3：Train**

```bash
cd final_project && python -m models.bert_finetune 2>&1 | tee models/bert_train.log
```
Expected: train for up to 5 epochs, early-stop on val macro-F1; checkpoint saved.
Time estimate：1–3 小時視硬體而定。

- [ ] **Step 4：Sanity check — run on val**

```bash
# Inspect log — val macro-F1 should > TF-IDF baseline by 10+ points
grep "macro_f1" models/bert_train.log
```

- [ ] **Step 5：Commit**

```bash
git add final_project/models/bert_finetune.py \
        final_project/models/bert_config.yaml \
        final_project/models/bert_ckpt/best/ \
        final_project/models/bert_train.log
git commit -m "feat(final): fine-tune Chinese MacBERT for multi-label classification"
```

---

## Phase 4：評估 + 錯誤分析 (Step 7)

### Task 10：統一 metrics module

**Files:**
- Create: `final_project/eval/metrics.py`
- Create: `final_project/tests/test_metrics.py`

- [ ] **Step 1：Write test**

```python
# tests/test_metrics.py
from eval.metrics import compute_multilabel_metrics

def test_perfect_predictions():
    gold = [{"a"}, {"a", "b"}]
    pred = [{"a"}, {"a", "b"}]
    m = compute_multilabel_metrics(gold, pred)
    assert m["micro_f1"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["hamming"] == 0.0
```

- [ ] **Step 2：Implement**

```python
# eval/metrics.py
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, classification_report

def compute_multilabel_metrics(gold, pred, labels=None):
    mlb = MultiLabelBinarizer(classes=labels)
    G = mlb.fit_transform(gold)
    P = mlb.transform(pred)
    return {
        "micro_f1": f1_score(G, P, average="micro", zero_division=0),
        "macro_f1": f1_score(G, P, average="macro", zero_division=0),
        "hamming": hamming_loss(G, P),
        "per_class": classification_report(G, P, target_names=list(mlb.classes_),
                                            zero_division=0, output_dict=True),
    }
```

- [ ] **Step 3：Commit**

```bash
git add final_project/eval/metrics.py final_project/tests/test_metrics.py
git commit -m "feat(final): unified multi-label metrics"
```

---

### Task 11：跑三法 + 產出比較表

**Files:**
- Create: `final_project/eval/run_all.py`
- Create: `final_project/eval/results.json`
- Create: `final_project/eval/comparison_table.md`

- [ ] **Step 1：Implement `eval/run_all.py`**

對 `test.jsonl` 跑 rule / tfidf_lr / tfidf_svm / bert 四個 classifier，收集 metrics，並額外跑 `aigrant` OOD 資料。

```python
"""Run all classifiers on in-domain test + OOD (aigrant), dump results."""
import json, pickle, yaml
from pathlib import Path
from datasets import load_dataset
from scripts.auto_labeler import load_ontology, map_citations_to_issues
from scripts.citation_extractor import extract_citations
from baselines.rule import RuleClassifier
from eval.metrics import compute_multilabel_metrics

# ---------- load in-domain test ----------
test_rows = [json.loads(l) for l in Path("data/test.jsonl").read_text().splitlines()]
X_id = [r["story"] for r in test_rows]
y_id = [set(r["labels"]) for r in test_rows]

# ---------- load OOD aigrant (truncate content to 1500 chars) ----------
ontology = load_ontology()
ds_ood = load_dataset("aigrant/taiwan-ly-law-research")
ood_rows = []
for r in ds_ood["train"]:
    rl = r.get("related_laws", "") or ""
    laws = [x.strip() for x in rl.split(";") if x.strip()]
    # map law NAME (not article) → collect issues whose any law matches
    issue_ids = set()
    for l in laws:
        for issue in ontology:
            if any(pair[0] == l for pair in issue["laws"]):
                issue_ids.add(issue["id"])
    if not issue_ids: continue
    content = (r.get("title", "") + "。" + (r.get("content", "") or ""))[:1500]
    ood_rows.append({"story": content, "labels": issue_ids})
X_ood = [r["story"] for r in ood_rows]
y_ood = [r["labels"] for r in ood_rows]
print(f"OOD rows: {len(ood_rows)}")

# ---------- classifiers ----------
results = {}

# rule
rule = RuleClassifier(ontology)
results["rule_id"]  = compute_multilabel_metrics(y_id, rule.predict_batch(X_id))
results["rule_ood"] = compute_multilabel_metrics(y_ood, rule.predict_batch(X_ood))

# tfidf + lr / svm
import jieba
jieba.load_userdict("baselines/custom_jieba_dict.txt")
def tok_batch(texts): return [" ".join(jieba.cut(t)) for t in texts]

for k in ["lr", "svm"]:
    bundle = pickle.load(open(f"models/tfidf_{k}.pkl", "rb"))
    Xv_id = bundle["vec"].transform(tok_batch(X_id))
    Xv_ood = bundle["vec"].transform(tok_batch(X_ood))
    P_id = bundle["clf"].predict(Xv_id)
    P_ood = bundle["clf"].predict(Xv_ood)
    preds_id = [set(bundle["mlb"].classes_[p.astype(bool)].tolist()) for p in P_id]
    preds_ood = [set(bundle["mlb"].classes_[p.astype(bool)].tolist()) for p in P_ood]
    results[f"tfidf_{k}_id"]  = compute_multilabel_metrics(y_id, preds_id)
    results[f"tfidf_{k}_ood"] = compute_multilabel_metrics(y_ood, preds_ood)

# bert
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("models/bert_ckpt/best")
model = AutoModelForSequenceClassification.from_pretrained("models/bert_ckpt/best")
label_classes = json.loads(Path("models/bert_ckpt/label_classes.json").read_text())
model.eval()

def predict_bert(texts, batch=16):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], truncation=True, padding=True,
                      max_length=512, return_tensors="pt")
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for p in probs:
                preds.append(set(lc for j, lc in enumerate(label_classes) if p[j] > 0.5))
    return preds

results["bert_id"]  = compute_multilabel_metrics(y_id, predict_bert(X_id))
results["bert_ood"] = compute_multilabel_metrics(y_ood, predict_bert(X_ood))

Path("eval/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

# ---------- comparison table ----------
lines = ["# Model Comparison\n"]
lines.append("## In-Domain (test.jsonl)\n")
lines.append("| Method | Micro-F1 | Macro-F1 | Hamming |")
lines.append("|---|---|---|---|")
for k in ["rule", "tfidf_lr", "tfidf_svm", "bert"]:
    m = results[f"{k}_id"]
    lines.append(f"| {k} | {m['micro_f1']:.3f} | {m['macro_f1']:.3f} | {m['hamming']:.3f} |")
lines.append("\n## OOD (aigrant)\n")
lines.append("| Method | Micro-F1 | Macro-F1 | Hamming |")
lines.append("|---|---|---|---|")
for k in ["rule", "tfidf_lr", "tfidf_svm", "bert"]:
    m = results[f"{k}_ood"]
    lines.append(f"| {k} | {m['micro_f1']:.3f} | {m['macro_f1']:.3f} | {m['hamming']:.3f} |")

Path("eval/comparison_table.md").write_text("\n".join(lines))
print("Done → eval/results.json, eval/comparison_table.md")
```

- [ ] **Step 2：Run**

```bash
cd final_project && python -m eval.run_all
```

- [ ] **Step 3：Commit**

```bash
git add final_project/eval/run_all.py \
        final_project/eval/results.json \
        final_project/eval/comparison_table.md
git commit -m "feat(final): evaluate all 4 models on in-domain + OOD"
```

---

### Task 12：錯誤分析 Notebook

**Files:**
- Create: `final_project/eval/error_analysis.ipynb`

- [ ] **Step 1：Create notebook with sections**

```
1. Load results.json
2. Per-class F1 bar chart (BERT)
3. Confusion heatmap (top混淆對)
4. Long-tail 分析：frequency vs F1 scatter
5. Error case study：
   - 挑 5 個 FP（模型多預測的）
   - 挑 5 個 FN（模型漏預測的）
   - 討論原因（同義詞缺失？訓練樣本不足？議題混淆？）
6. OOD vs ID performance gap 分析
```

- [ ] **Step 2：Commit**

```bash
git add final_project/eval/error_analysis.ipynb
git commit -m "feat(final): error analysis notebook with case studies"
```

---

### Task 13：Demo Notebook

**Files:**
- Create: `final_project/demo.ipynb`

- [ ] **Step 1：Build notebook**

```
1. Load 3 classifiers (rule / tfidf_svm / bert)
2. Load tw-processed-law-article lookup table
3. Input widget：textarea for 自由文字
4. For each classifier：predict labels + confidence
5. Display 3-column comparison table
6. For BERT predictions：lookup related 法條原文 → 顯示
7. Sample queries 預設：
   - 「我被告妨害名譽 但是 threads 上的人是冒名的 怎麼辦」
   - 「朋友酒駕被抓吐氣 0.3 會怎樣」
   - 「做 3 年突然被解僱沒資遣費」
8. 免責聲明（僅供參考、不構成法律諮詢）
```

- [ ] **Step 2：Commit**

```bash
git add final_project/demo.ipynb
git commit -m "feat(final): interactive demo notebook with 3-model comparison"
```

---

## Phase 5：報告 & Slides

### Task 14：Written Report（3–6 頁 PDF）

**Files:**
- Create: `final_project/report/report.md`
- Create: `final_project/report/figs/` (copy charts from notebooks)
- Create: `final_project/report/report.pdf`

- [ ] **Step 1：Write `report/report.md`（章節架構）**

```
1. 摘要 (Abstract) — 200 字
2. 引言 — 問題 / 目標使用者 / 相關競品 — 0.5 頁
3. 資料集 — 來源 / 授權 / 探索統計 — 1 頁
   • 引用 DATA_USABILITY.md 的表格
4. 方法
   4.1 議題 Ontology 建構 — 0.5 頁
   4.2 Rule-based / TF-IDF+SVM / BERT — 1 頁
5. 實驗 & 結果
   5.1 in-domain 比較表 + macro/micro F1 討論 — 0.5 頁
   5.2 類別不平衡分析 — 0.5 頁
   5.3 OOD 泛化分析 — 0.5 頁
6. 錯誤分析 + Case study — 1 頁
7. 實務應用 + 限制 — 0.5 頁
8. 結論 — 0.3 頁
9. References
```

- [ ] **Step 2：Export to PDF**

```bash
cd final_project/report
pandoc report.md -o report.pdf \
  --pdf-engine=xelatex \
  -V CJKmainfont="PingFang TC" \
  -V geometry:margin=2.5cm
```

- [ ] **Step 3：Commit**

```bash
git add final_project/report/
git commit -m "docs(final): written report draft"
```

---

### Task 15：Oral Presentation Slides

**Files:**
- Create: `final_project/report/slides.md`（Marp 格式）
- Create: `final_project/report/slides.pdf`

- [ ] **Step 1：Write slides**

10–12 分鐘講稿，~15–18 張投影片：
1. 封面
2. 使用者故事（threads 冒名案例）
3. 問題 & 目標使用者
4. 資料探索：發現 / 統計
5. 議題 Ontology 建構過程
6. Pipeline 流程圖
7. 三法簡介
8. In-domain 比較表
9. OOD 結果
10. Per-class F1 分析
11. Case study
12. 限制 & future work
13. Demo 現場跑
14. Q&A

- [ ] **Step 2：Export PDF**

```bash
npx @marp-team/marp-cli@latest slides.md -o slides.pdf
```

- [ ] **Step 3：Commit**

```bash
git add final_project/report/slides.md final_project/report/slides.pdf
git commit -m "docs(final): oral presentation slides"
```

---

### Task 16：Demo Recording（防當機備案）

**Files:**
- Create: `final_project/report/demo_video.mp4`

- [ ] **Step 1：Record 2-minute screen recording of demo.ipynb**

Contents:
- 啟動 notebook
- 跑 3 個預設 query
- 展示條文原文 lookup
- 展示 BERT 比其他模型多抓到的議題

- [ ] **Step 2：Commit**（或放 YouTube 不 commit）

---

## Phase 6：收尾

### Task 17：End-to-end 重現測試

- [ ] **Step 1：Fresh clone test**

在新資料夾重新 clone，跑完整 flow：
```bash
git clone <repo> test_reproduce
cd test_reproduce/final_project
uv pip install -e .
pytest tests/ -v
python -m scripts.analyze_pair_freq
python -m scripts.build_labeled_dataset
python -m scripts.split_data
python -m baselines.tfidf_svm --model lr
python -m eval.run_all
```

確認結果與原 repo 一致（metrics 差異 <1%）。

- [ ] **Step 2：Fix any environment issues, update README**

- [ ] **Step 3：Final commit**

```bash
git commit -am "docs(final): reproducibility validated"
```

---

## 附錄：時程預估

| Phase | Tasks | 預估時數（自己跑） |
|---|---|---|
| Phase 0 | Task 0 | 1 hr |
| Phase 1 | Task 1–5 | 5–6 hr |
| Phase 1.5（可選） | Task 5a–5d | 2–4 hr（依擴充範圍）|
| Phase 1（續） | Task 6 | 0.5 hr |
| Phase 2 | Task 7–8 | 3–4 hr |
| Phase 3 | Task 9 | 3–5 hr（含訓練時間） |
| Phase 4 | Task 10–13 | 4–6 hr |
| Phase 5 | Task 14–16 | 4–6 hr |
| Phase 6 | Task 17 | 1 hr |
| **Total** | | **24–34 小時（分 8–11 個工作天）** |

---

## 附錄：Commit 命名規範

```
feat(final): <新功能描述>
fix(final): <修 bug 描述>
docs(final): <文件相關>
test(final): <新增 / 修改測試>
chore(final): <雜項，例如依賴更新>
```

每個 Task 至少 1 commit，複雜 Task 可拆多個 sub-commit（例：Task 3 的 stub generation、人工精修、validator 分 3 commit）。

---

## 附錄：關鍵風險清單 & 對策

| 風險 | 機率 | 影響 | 對策 |
|---|---|---|---|
| Step 3 ontology 設計耗時超出預期 | 高 | 中 | 限 1 天，不完美就先 freeze 跑 pipeline，Phase 5 再回補 |
| Step 4 自動打標後資料 <2,000 筆 | 中 | 高 | 放寬過濾條件；或合併 aigrant 部分進訓練 |
| BERT 在 val 分數比 TF-IDF 差 | 低 | 高 | lr 降到 1e-5，freeze bottom layers；如還是差，報告誠實寫「小資料下 BERT 優勢有限」|
| aigrant OOD 分數極差 | 高 | 低 | 預期會差，當 limitation 寫 |
| Pandoc 中文 PDF 匯出問題 | 中 | 低 | 備案：用 Typst / LaTeX / 直接用 Obsidian export |
| 期末時程壓縮 | 中 | 中 | SHOULD HAVE (錄影、OOD、詞典精修) 優先犧牲 |
