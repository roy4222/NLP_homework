"""Audit regex label extraction quality on random sample — does it produce usable labels?"""
from datasets import load_dataset
import re, random, json
from collections import Counter

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
    "建築法", "廢棄物清理法", "住宅法", "教師法", "大學法",
    "醫療法", "再生能源發展條例", "國民教育法",
]
LAW_NAMES.sort(key=len, reverse=True)  # longer first to avoid prefix matches
LAW_PAT = re.compile(
    r"(" + "|".join(LAW_NAMES) + r")"
    r"\s*第\s*"
    r"([0-9零一二三四五六七八九十百千]+)"
    r"\s*條"
    r"(?:\s*之\s*([0-9一二三四五六七八九十]+))?"
)

def normalize_law(name):
    if name == "中華民國刑法": return "刑法"
    if name == "勞基法": return "勞動基準法"
    if name == "性別平等工作法": return "性別工作平等法"
    return name

def cn2arabic(s):
    m = {"零":0,"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}
    if all(c.isdigit() for c in s):
        return int(s)
    # simple parse for 十/百 levels
    n = 0; cur = 0
    for c in s:
        v = m.get(c)
        if v is None: return s
        if v >= 10:
            if cur == 0: cur = 1
            n += cur * v
            cur = 0
        else:
            cur = v
    n += cur
    return n

print("Loading tw-legal-synthetic-qa...")
ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
all_rows = list(ds["train"]) + list(ds["test"])
print(f"Total: {len(all_rows)}")

# Full-pass stats with cleaned regex
rows_with_cite = 0
pairs_per_row = []
law_counter = Counter()
pair_counter = Counter()
for r in all_rows:
    text = r["messages"][0]["content"] + "\n" + r["messages"][1]["content"]
    matches = LAW_PAT.findall(text)
    pairs = set()
    for law, num, _ in matches:
        law = normalize_law(law)
        num = cn2arabic(num)
        pairs.add((law, num))
    if pairs:
        rows_with_cite += 1
        pairs_per_row.append(len(pairs))
        for p in pairs:
            pair_counter[p] += 1
            law_counter[p[0]] += 1

print(f"\n=== POST-CLEANUP NUMBERS (dedup per row, normalized) ===")
print(f"Rows with ≥1 usable citation: {rows_with_cite} ({rows_with_cite/len(all_rows)*100:.1f}%)")
print(f"Avg unique citations per row: {sum(pairs_per_row)/len(pairs_per_row):.2f}")
print(f"Median: {sorted(pairs_per_row)[len(pairs_per_row)//2]}")
print(f"Unique (law, article) pairs: {len(pair_counter)}")
print(f"Unique law names: {len(law_counter)}")
print(f"\nTop 15 laws (by row count):")
for law, cnt in law_counter.most_common(15):
    print(f"  {law}: {cnt}")
print(f"\nTop 30 (law, article) pairs:")
for (law, num), cnt in pair_counter.most_common(30):
    print(f"  {law} §{num}: {cnt}")

# pair freq distribution
fdist = Counter()
for c in pair_counter.values():
    if c >= 100: fdist["≥100"] += 1
    elif c >= 50: fdist["50–99"] += 1
    elif c >= 20: fdist["20–49"] += 1
    elif c >= 10: fdist["10–19"] += 1
    elif c >= 5:  fdist["5–9"] += 1
    elif c >= 2:  fdist["2–4"] += 1
    else:         fdist["1"] += 1
print(f"\nPair frequency distribution: {dict(fdist)}")

# Random sample audit
print(f"\n=== RANDOM SAMPLE OF 10 EXTRACTIONS (for manual eyeball) ===")
random.seed(42)
sampled = random.sample([r for r in all_rows if LAW_PAT.search(r["messages"][0]["content"]+r["messages"][1]["content"])], 10)
for i, r in enumerate(sampled):
    user_msg = r["messages"][0]["content"]
    asst_msg = r["messages"][1]["content"]
    matches = set()
    for law, num, _ in LAW_PAT.findall(user_msg + "\n" + asst_msg):
        matches.add((normalize_law(law), cn2arabic(num)))
    print(f"\n--- sample #{i+1} ---")
    print(f"USER (first 200 chars): {user_msg[:200].strip()}...")
    print(f"ASSISTANT EXTRACTED: {sorted(matches)}")
    print(f"ASSISTANT (first 200 chars): {asst_msg[:200].strip()}...")
