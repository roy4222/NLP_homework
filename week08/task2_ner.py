"""Task 2: spaCy 中文 NER 抓人名與地名"""
import spacy

sample_text = """
「昨晚八點，鴻海精密的前董事長郭台明在台北信義區舉辦了一場科技論壇。
會中他提到了頻果公司的創辦人賈伯斯對全球產業的影響。
與此同時，蔡英蚊也在臉書上轉發了相關新聞。
雖然論壇地點原本預計在香港的銅羅灣舉行，但因為行程調整，最後改到了台灣。
不少來自美國加洲的工程師也透過視訊參與了這場由聯合國教科文組織贊助的活動，
時間定在2024年三月正式發布報告。」
"""

nlp = spacy.load("zh_core_web_sm")
doc = nlp(sample_text)

print("=" * 60)
print("Task 2: spaCy Chinese NER")
print("=" * 60)
print("所有辨識到的實體：")
for ent in doc.ents:
    print(f"  {ent.text:<10} {ent.label_:<8} ({spacy.explain(ent.label_)})")

people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
locations = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]

print("-" * 60)
print(f"找到的人名 (PERSON)  : {people}")
print(f"找到的地名 (GPE/LOC) : {locations}")
