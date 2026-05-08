"""Task 3: Levenshtein 實體正規化 + 文章替換"""
import re
from rapidfuzz.distance import Levenshtein

entity_list = [
    "台積電", "台灣積體電路製造公司", "台積電股份有限公司", "台機電",
    "鴻海", "鴻海精密", "鴻海科技集團", "紅海精密",
    "Google", "GoogIe", "Google Inc.", "谷歌",
    "國立臺灣大學", "台灣大學", "台大", "國立台灣大學",
]

messy_text = """
根據今日財報分析，台灣積體電路製造公司 在本季表現亮眼，
而另一家半導體大廠 台機電 也表示產能滿載。
與此同時，鴻海精密 與其夥伴 鴻海科技集團 宣布了新的合作計畫，
但在網路論壇上，部分網友卻誤植為 紅海精密。

學術界方面，國立臺灣大學 發表了最新的 AI 研究成果，
參與研究的學生不少來自 台灣大學 與 台大，
這項研究也獲得了美國科技巨頭 GoogIe 的技術支援，
雖然在合約中對方的正式名稱標註為 Google Inc.。
"""

standard_list = [
    "台灣積體電路製造股份有限公司",
    "鴻海精密工業股份有限公司",
    "Google LLC",
    "國立臺灣大學",
]

alias_to_standard = {
    "台積電": "台灣積體電路製造股份有限公司",
    "台灣積體電路製造公司": "台灣積體電路製造股份有限公司",
    "台積電股份有限公司": "台灣積體電路製造股份有限公司",
    "台機電": "台灣積體電路製造股份有限公司",
    "鴻海": "鴻海精密工業股份有限公司",
    "鴻海精密": "鴻海精密工業股份有限公司",
    "鴻海科技集團": "鴻海精密工業股份有限公司",
    "紅海精密": "鴻海精密工業股份有限公司",
    "Google": "Google LLC",
    "GoogIe": "Google LLC",
    "Google Inc.": "Google LLC",
    "谷歌": "Google LLC",
    "國立臺灣大學": "國立臺灣大學",
    "台灣大學": "國立臺灣大學",
    "台大": "國立臺灣大學",
    "國立台灣大學": "國立臺灣大學",
}

def normalize(entity, standards, threshold=0.6):
    scores = [(s, Levenshtein.normalized_similarity(entity, s)) for s in standards]
    best, score = max(scores, key=lambda x: x[1])
    if score >= threshold:
        return best, score, "Levenshtein"
    if entity in alias_to_standard:
        a = alias_to_standard[entity]
        return a, Levenshtein.normalized_similarity(entity, a), "alias"
    return None, score, "skip"

print("=" * 70)
print("Task 3: Entity Normalization with Levenshtein Distance")
print("=" * 70)
print("標準名稱表：")
for s in standard_list:
    print(" -", s)

print("\n別名 → 標準對應結果：")
mapping = {}
for e in entity_list:
    s, sc, m = normalize(e, standard_list, 0.6)
    if s is not None:
        mapping[e] = s
    print(f"  {e:<11} → {s}   [score={sc:.2f}, {m}]")

# 一次掃完原文，避免「替換後字串再被局部替換」(例如鴻海→鴻海精密工業..精密工業..)
pattern = re.compile("|".join(re.escape(a) for a in sorted(mapping, key=len, reverse=True)))
cleaned_text = pattern.sub(lambda m: mapping[m.group(0)], messy_text)

print("\n" + "=" * 70)
print("清理後文章：")
print("=" * 70)
print(cleaned_text)
