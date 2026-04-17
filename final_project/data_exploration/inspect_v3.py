"""v3: inspect new candidate datasets."""
from datasets import load_dataset
import re
from collections import Counter

law_pat = re.compile(
    r"(?:《\s*)?"
    r"((?:中華民國\s*)?(?:刑法|民法|勞動基準法|勞基法|性別工作平等法|性別平等工作法|就業服務法|"
    r"勞工保險條例|勞工退休金條例|勞資爭議處理法|職業安全衛生法|就業保險法|"
    r"民事訴訟法|刑事訴訟法|行政訴訟法|行政程序法|家事事件法|家庭暴力防治法|"
    r"道路交通管理處罰條例|毒品危害防制條例|公司法|專利法|商標法|著作權法|"
    r"消費者保護法|個人資料保護法|公平交易法|證券交易法|票據法|保險法|所得稅法))"
    r"(?:\s*》)?"
    r"\s*第\s*"
    r"([0-9零一二三四五六七八九十百千]+)"
    r"\s*條"
)

def analyze(name, rows_iter, get_text, label="text"):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    total = 0
    with_cite = 0
    law_counter = Counter()
    for r in rows_iter:
        total += 1
        txt = get_text(r)
        matches = law_pat.findall(txt)
        if matches:
            with_cite += 1
            for law, _ in matches:
                law_counter[law] += 1
    print(f"  total rows: {total}")
    print(f"  rows with citation: {with_cite} ({with_cite/total*100:.1f}%)")
    print(f"  top 10 laws:")
    for law, cnt in law_counter.most_common(10):
        print(f"    {law}: {cnt}")

# 1. tw-legal-qa-3M — high priority
print("Loading tw-legal-qa-3M...")
try:
    ds = load_dataset("lianghsun/tw-legal-qa-3M")
    print(f"Splits: {ds}")
    for split in ds.keys():
        first = ds[split][0]
        print(f"\n[{split}] keys: {list(first.keys())}")
        print(f"Example text (first 600 chars):")
        print(first.get("text", first.get("messages", ""))[:600] if isinstance(first.get("text", first.get("messages", "")), str) else str(first)[:600])
    # analyze
    if "train" in ds:
        analyze("tw-legal-qa-3M (train)", ds["train"], lambda r: r.get("text", ""))
except Exception as e:
    print(f"FAIL: {e}")

# 2. tw-legal-qa-chat — small supplement
print("\n\nLoading tw-legal-qa-chat...")
try:
    ds = load_dataset("lianghsun/tw-legal-qa-chat")
    print(f"Splits: {ds}")
    for split in ds.keys():
        first = ds[split][0]
        print(f"\n[{split}] keys: {list(first.keys())}")
        out = first.get("output", "")
        inp = first.get("input", "")
        print(f"Example input (first 300 chars):\n  {inp[:300]}")
        print(f"Example output (first 500 chars):\n  {out[:500]}")
    if "train" in ds:
        analyze("tw-legal-qa-chat (train)", ds["train"],
                lambda r: (r.get("input", "") + "\n" + r.get("output", "")))
except Exception as e:
    print(f"FAIL: {e}")

# 3. aigrant/taiwan-ly-law-research
print("\n\nLoading aigrant/taiwan-ly-law-research...")
try:
    ds = load_dataset("aigrant/taiwan-ly-law-research")
    print(f"Splits: {ds}")
    first = ds["train"][0] if "train" in ds else list(ds.values())[0][0]
    print(f"keys: {list(first.keys())}")
    print(f"title: {first.get('title', '')[:200]}")
    print(f"related_laws: {first.get('related_laws', '')[:200]}")
    print(f"content (first 400 chars): {str(first.get('content', ''))[:400]}")
    # distribution of related_laws
    law_names = Counter()
    n_laws_per_row = Counter()
    for r in ds["train"]:
        rl = r.get("related_laws", "") or ""
        names = [x.strip() for x in rl.split(";") if x.strip()]
        n_laws_per_row[len(names)] += 1
        for n in names:
            law_names[n] += 1
    print(f"\nRelated laws per row distribution: {dict(n_laws_per_row)}")
    print(f"Unique laws referenced: {len(law_names)}")
    print(f"Top 20 laws:")
    for n, c in law_names.most_common(20):
        print(f"  {n}: {c}")
except Exception as e:
    print(f"FAIL: {e}")

# 4. tw-legal-methodology — lower priority but check
print("\n\nLoading tw-legal-methodology...")
try:
    ds = load_dataset("lianghsun/tw-legal-methodology")
    print(f"Splits: {ds}")
    for i in range(3):
        print(f"\n[row {i}]: {ds['train'][i].get('text', '')[:400]}")
except Exception as e:
    print(f"FAIL: {e}")

# 5. tw-legal-news-24M — curious
print("\n\nLoading tw-legal-news-24M...")
try:
    ds = load_dataset("lianghsun/tw-legal-news-24M")
    print(f"Splits: {ds}")
    first = ds["train"][0] if "train" in ds else list(ds.values())[0][0]
    print(f"keys: {list(first.keys())}")
    for k, v in first.items():
        s = str(v)[:200]
        print(f"  {k}: {s}")
except Exception as e:
    print(f"FAIL: {e}")
