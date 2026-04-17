"""Inspect erhwenkuo/wikinews-zhtw dataset for A3 K-Means assignment."""
from datasets import load_dataset
from collections import Counter
import json

print("Loading erhwenkuo/wikinews-zhtw...")
ds = load_dataset("erhwenkuo/wikinews-zhtw")
print(f"\nSplits: {ds}")

for split in ds.keys():
    print(f"\n=== {split} ({len(ds[split])} rows) ===")
    row = ds[split][0]
    print(f"Keys: {list(row.keys())}")
    for k, v in row.items():
        preview = str(v)[:300]
        print(f"  {k}: {preview}")

# text length stats
split = list(ds.keys())[0]
rows = ds[split]
text_field = None
for candidate in ["text", "content", "article", "body"]:
    if candidate in rows[0]:
        text_field = candidate
        break
print(f"\nDetected text field: {text_field}")

if text_field:
    lengths = [len(r[text_field]) for r in rows]
    print(f"Text length — min: {min(lengths)}, max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.0f}")

# check if labeled
label_field = None
for candidate in ["label", "category", "topic", "class"]:
    if candidate in rows[0]:
        label_field = candidate
        break
print(f"Detected label field: {label_field}")

if label_field:
    labels = Counter(r[label_field] for r in rows)
    print(f"Label distribution ({len(labels)} classes):")
    for k, c in labels.most_common():
        print(f"  {k}: {c}")
else:
    print("\n⚠️ No category label → pure unsupervised task (K-Means clusters need manual naming)")

# sample 3 full records
print("\n=== 3 sample records (full) ===")
for i in range(3):
    r = rows[i]
    print(f"\n--- record {i} ---")
    for k, v in r.items():
        s = str(v)
        print(f"  {k}: {s[:500] if len(s) > 500 else s}")
