"""Audit aigrant dataset: label overlap with synthetic-qa, feasibility as OOD test."""
from datasets import load_dataset
from collections import Counter
import re

ds1 = load_dataset("lianghsun/tw-legal-synthetic-qa")
ds2 = load_dataset("aigrant/taiwan-ly-law-research")

LAW_NAMES = [
    "中華民國刑法","刑法","民法","勞動基準法","性別工作平等法","就業服務法",
    "勞工保險條例","勞工退休金條例","勞資爭議處理法","職業安全衛生法","就業保險法",
    "民事訴訟法","刑事訴訟法","行政訴訟法","行政程序法","家事事件法","家庭暴力防治法",
    "道路交通管理處罰條例","毒品危害防制條例","公司法","專利法","商標法","著作權法",
    "消費者保護法","個人資料保護法","公平交易法","證券交易法","票據法","保險法",
    "所得稅法","兒童及少年福利與權益保障法","建築法","廢棄物清理法","住宅法","教師法",
    "大學法","醫療法","再生能源發展條例","國民教育法",
]
def normalize(n):
    return "刑法" if n == "中華民國刑法" else n

# Laws appearing in synthetic-qa (what we train on)
LAW_PAT = re.compile(r"(" + "|".join(sorted(LAW_NAMES, key=len, reverse=True)) + r")\s*第\s*[0-9零一二三四五六七八九十百千]+\s*條")
synth_laws = Counter()
for r in list(ds1["train"]) + list(ds1["test"]):
    txt = r["messages"][0]["content"] + "\n" + r["messages"][1]["content"]
    for law in LAW_PAT.findall(txt):
        synth_laws[normalize(law)] += 1

# Laws in aigrant
aigrant_laws = Counter()
aigrant_rows_with_label = 0
for r in ds2["train"]:
    rl = r.get("related_laws", "") or ""
    laws = [x.strip() for x in rl.split(";") if x.strip()]
    if laws:
        aigrant_rows_with_label += 1
    for l in laws:
        aigrant_laws[l] += 1

print(f"Synthetic-QA unique laws: {len(synth_laws)} (only normalized names)")
print(f"Aigrant unique laws: {len(aigrant_laws)}")
print(f"Aigrant rows with ≥1 label: {aigrant_rows_with_label}/{len(ds2['train'])}")

# Overlap
synth_set = set(synth_laws.keys())
aigrant_set = set(aigrant_laws.keys())
overlap = synth_set & aigrant_set
print(f"\nLaw-name overlap: {len(overlap)} laws")
print(f"In both: {sorted(overlap)}")
print(f"\nSynth-only (train-only): {sorted(synth_set - aigrant_set)}")
aigrant_only_sample = sorted(aigrant_set - synth_set, key=lambda k: -aigrant_laws[k])[:30]
print(f"\nAigrant-only (top 30 by freq): {aigrant_only_sample}")

# Usable aigrant rows: those with at least 1 label that overlaps synth
usable = 0
for r in ds2["train"]:
    rl = r.get("related_laws", "") or ""
    laws = [x.strip() for x in rl.split(";") if x.strip()]
    if any(l in overlap for l in laws):
        usable += 1
print(f"\n=== FEASIBILITY AS OOD TEST ===")
print(f"Aigrant rows with at least 1 label shared with synthetic-qa label space: {usable}")
print(f"→ This is the max usable for OOD evaluation (under shared-label constraint)")

# Content length check
lens = []
for r in ds2["train"]:
    c = r.get("content", "") or ""
    lens.append(len(c))
print(f"\nAigrant content length — min: {min(lens)}, max: {max(lens)}, mean: {sum(lens)/len(lens):.0f}")
print(f"Content >2000 chars: {sum(1 for l in lens if l > 2000)} ({sum(1 for l in lens if l > 2000)/len(lens)*100:.1f}%)")
print(f"  ⚠️ BERT max length 512 tokens ≈ 1000-1500 chars → most aigrant content needs truncation")
