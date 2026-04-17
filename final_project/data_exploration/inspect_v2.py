"""Second pass: better regex, scan FULL dataset, check other datasets too."""
from datasets import load_dataset
import re, json
from collections import Counter

# Better regex: catch both Arabic and Chinese numerals, any law name ending in 法/條例
law_pat = re.compile(
    r"(?:《\s*)?"
    r"([\u4e00-\u9fff]{2,12}?(?:法(?:規|典|典?)?|條例|通則|辦法|細則|準則))"
    r"(?:\s*》)?"
    r"\s*第\s*"
    r"([0-9零一二三四五六七八九十百千]+)"
    r"\s*條"
    r"(?:\s*之\s*([0-9一二三四五六七八九十]+))?"
)

# Labor-related law names
LABOR = {"勞動基準法", "勞基法", "性別工作平等法", "性別平等工作法", "性平法",
         "就業服務法", "勞工保險條例", "勞工退休金條例", "勞資爭議處理法",
         "職業安全衛生法", "就業保險法", "勞工職業災害保險及保護法"}

print("=" * 70)
print("tw-legal-synthetic-qa — FULL scan")
print("=" * 70)
ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
full = list(ds["train"]) + list(ds["test"])
print(f"Total rows: {len(full)}")

law_counter = Counter()
pair_counter = Counter()
rows_with_citation = 0
rows_labor = 0
hit_dist = Counter()
for r in full:
    user = r["messages"][0]["content"]
    asst = r["messages"][1]["content"]
    text = user + "\n" + asst
    matches = law_pat.findall(text)
    hit_dist[len(matches)] += 1
    if matches:
        rows_with_citation += 1
    has_labor = False
    for law, num, sub in matches:
        law_counter[law] += 1
        pair_counter[(law, num)] += 1
        if law in LABOR or any(k in law for k in ["勞動", "勞工", "勞資", "性別工作", "性別平等工作", "就業服務", "就業保險", "職業安全"]):
            has_labor = True
    if has_labor:
        rows_labor += 1

print(f"\nRows with >=1 law citation: {rows_with_citation} ({rows_with_citation/len(full)*100:.1f}%)")
print(f"Rows with labor-law citation: {rows_labor} ({rows_labor/len(full)*100:.1f}%)")
print(f"\nDistribution of hits per row (top 10):")
for h, c in sorted(hit_dist.items())[:10]:
    print(f"  {h} hits: {c} rows")

print(f"\nTop 30 laws cited:")
for law, cnt in law_counter.most_common(30):
    print(f"  {law}: {cnt}")

print(f"\nUnique (law, article#) pairs: {len(pair_counter)}")
print(f"Top 30 pairs:")
for (law, num), cnt in pair_counter.most_common(30):
    print(f"  {law} 第{num}條: {cnt}")

# Distribution of pair frequencies — key for classification feasibility
freqs = Counter()
for cnt in pair_counter.values():
    if cnt >= 50: freqs["≥50"] += 1
    elif cnt >= 20: freqs["≥20"] += 1
    elif cnt >= 10: freqs["≥10"] += 1
    elif cnt >= 5: freqs["≥5"] += 1
    elif cnt >= 2: freqs["≥2"] += 1
    else: freqs["=1"] += 1
print(f"\nPair frequency distribution: {dict(freqs)}")

# Aggregate at law-name level
print(f"\nLaw-name level: {len(law_counter)} unique laws")
laws_with_threshold = {k: v for k, v in law_counter.items() if v >= 30}
print(f"Laws with ≥30 citations: {len(laws_with_threshold)}")
print(f"Laws with ≥100 citations: {sum(1 for v in law_counter.values() if v >= 100)}")

# Check tw-legal-nlp
print("\n" + "=" * 70)
print("tw-legal-nlp — for completeness")
print("=" * 70)
try:
    ds2 = load_dataset("lianghsun/tw-legal-nlp")
    print(ds2)
    for r in ds2["train"].select(range(3)):
        print(json.dumps(r, ensure_ascii=False)[:500])
    task_counter = Counter(r["task"] for r in ds2["train"])
    print(f"Task distribution: {dict(task_counter)}")
except Exception as e:
    print(f"Failed: {e}")

# Check benchmark
print("\n" + "=" * 70)
print("tw-legal-benchmark-v1 — for completeness")
print("=" * 70)
try:
    ds3 = load_dataset("lianghsun/tw-legal-benchmark-v1")
    print(ds3)
    for r in ds3["train"].select(range(2)):
        print(json.dumps(r, ensure_ascii=False)[:500])
except Exception as e:
    print(f"Failed: {e}")

# Processed-law-article: check size & structure
print("\n" + "=" * 70)
print("tw-processed-law-article — sample only (large dataset)")
print("=" * 70)
try:
    ds4 = load_dataset("lianghsun/tw-processed-law-article", split="train", streaming=True)
    samples = []
    for i, r in enumerate(ds4):
        samples.append(r)
        if i >= 3: break
    for s in samples:
        print(json.dumps(s, ensure_ascii=False)[:400])
        print("---")
    # Check if any "level" distribution available
    # stream first 10k to get level distribution
    levels = Counter()
    names = Counter()
    for i, r in enumerate(load_dataset("lianghsun/tw-processed-law-article", split="train", streaming=True)):
        levels[r.get("level", "")] += 1
        names[r.get("name", "")] += 1
        if i >= 20000: break
    print(f"\nSampled 20k rows:")
    print(f"  levels: {dict(levels)}")
    print(f"  unique law names: {len(names)}")
    print(f"  top 20 laws by article count:")
    for n, c in names.most_common(20):
        print(f"    {n}: {c}")
    # labor laws
    print(f"\n  labor-related laws in sample:")
    for n, c in names.most_common():
        if any(k in n for k in ["勞動", "勞工", "勞資", "性別工作", "性別平等工作", "就業服務", "就業保險", "職業安全"]):
            print(f"    {n}: {c}")
except Exception as e:
    import traceback; traceback.print_exc()
