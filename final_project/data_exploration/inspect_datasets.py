"""Inspect all 4 candidate datasets to decide final project scope."""
from datasets import load_dataset
import json, re
from collections import Counter

print("=" * 70)
print("1. tw-legal-synthetic-qa")
print("=" * 70)
ds = load_dataset("lianghsun/tw-legal-synthetic-qa")
print(ds)
for split in ds.keys():
    print(f"  {split}: {len(ds[split])}")

print("\n--- first 2 rows (train) ---")
for i, row in enumerate(ds["train"].select(range(2))):
    print(f"\n[row {i}] keys: {list(row.keys())}")
    print(json.dumps(row, ensure_ascii=False, indent=2)[:2000])

print("\n--- sample user prompt lengths ---")
lengths = [len(r["messages"][0]["content"]) for r in ds["train"].select(range(500))]
print(f"  user msg chars: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")

print("\n--- regex-extract law citations from assistant (500 sample) ---")
law_pat = re.compile(r"《?([^《》〈〉，。\s：:「」()（）]{2,15}?法|[^《》，。\s：:「」()（）]{2,15}?條例|[^《》，。\s：:「」()（）]{2,15}?法典)》?\s*第\s*([0-9〇一二三四五六七八九十百千零]+)\s*條(?:之\s*[0-9一二三四五六七八九十]+)?")
law_pat2 = re.compile(r"(刑法|民法|憲法|勞動基準法|勞基法|性別工作平等法|性平法|就業服務法|勞工保險條例|勞工退休金條例|公司法|專利法|商標法|著作權法|消費者保護法|個人資料保護法|道路交通管理處罰條例|家事事件法|家庭暴力防治法|家暴法|公平交易法|證券交易法|行政程序法|行政訴訟法|民事訴訟法|刑事訴訟法|票據法|保險法|勞資爭議處理法|職業安全衛生法|就業保險法|性別平等工作法)\s*第\s*([0-9〇一二三四五六七八九十百千零]+)\s*條(?:之\s*[0-9一二三四五六七八九十]+)?")
cite_counter = Counter()
law_counter = Counter()
hits_per_row = []
for r in ds["train"].select(range(500)):
    asst = r["messages"][1]["content"]
    matches = law_pat2.findall(asst)
    hits_per_row.append(len(matches))
    for law, num in matches:
        law_counter[law] += 1
        cite_counter[(law, num)] += 1
print(f"  sampled 500 rows")
print(f"  hits per row: mean={sum(hits_per_row)/len(hits_per_row):.2f}, 0-hit rows={sum(1 for h in hits_per_row if h==0)}")
print(f"  top 20 laws referenced:")
for law, cnt in law_counter.most_common(20):
    print(f"    {law}: {cnt}")
print(f"  top 20 (law, article) pairs:")
for (law, num), cnt in cite_counter.most_common(20):
    print(f"    {law} 第{num}條: {cnt}")

print("\n--- labor-law-only filter test ---")
labor_laws = ["勞動基準法", "勞基法", "性別工作平等法", "性平法", "性別平等工作法",
              "就業服務法", "勞工保險條例", "勞工退休金條例", "勞資爭議處理法",
              "職業安全衛生法", "就業保險法"]
labor_pat = re.compile("|".join(labor_laws))
labor_rows = 0
for r in ds["train"]:
    if labor_pat.search(r["messages"][1]["content"]) or labor_pat.search(r["messages"][0]["content"]):
        labor_rows += 1
print(f"  rows containing labor-law keywords in any message: {labor_rows} / {len(ds['train'])} ({labor_rows/len(ds['train'])*100:.1f}%)")
