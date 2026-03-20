"""
Question 01: Can the model correctly identify plurals?
使用 spaCy small model 判斷 "Three geese crossed the road" 的名詞單複數。
"""

import spacy
from enum import Enum


class Noun_number(Enum):
    SINGULAR = 1
    PLURAL = 2


# ── 講義原始版本（有 bug）──────────────────────────────────
def get_nouns_number(text, model, method="lemma"):
    """講義上的原始函式，morph 分支有兩個問題。"""
    nouns = []
    doc = model(text)
    for token in doc:
        if token.pos_ == "NOUN":
            if method == "lemma":
                if token.lemma_ != token.text:
                    nouns.append((token.text, Noun_number.PLURAL))
                else:
                    nouns.append((token.text, Noun_number.SINGULAR))
            elif method == "morph":
                # Bug 1: morph.get() 回傳 list（如 ['Sing']），不是字串
                # Bug 2: 判斷邏輯反了 — "Sing" 卻標成 PLURAL
                if token.morph.get("Number") == "Sing":
                    nouns.append((token.text, Noun_number.PLURAL))
                else:
                    nouns.append((token.text, Noun_number.SINGULAR))
    return nouns


if __name__ == "__main__":
    small_model = spacy.load("en_core_web_sm")
    text = "Three geese crossed the road"

    print("=== 講義原始程式碼 ===")
    print()

    nouns = get_nouns_number(text, small_model, "morph")
    print(f"morph 方法: {nouns}")

    nouns = get_nouns_number(text, small_model)
    print(f"lemma 方法: {nouns}")

    print()
    print("=== 回答 ===")
    print("First print (morph):  No — geese 被判成 SINGULAR，不正確")
    print("Second print (lemma): Yes — geese 被判成 PLURAL，正確")

    print()
    print("=== 原因分析 ===")
    print()

    doc = small_model(text)
    for token in doc:
        if token.pos_ == "NOUN":
            print(f"  {token.text}:")
            print(f"    lemma = {token.lemma_!r}")
            print(f"    morph.get('Number') = {token.morph.get('Number')!r}  (type: {type(token.morph.get('Number')).__name__})")
            print(f"    tag = {token.tag_}")

    print()
    print("morph 方法出錯的原因有兩個：")
    print("  1. morph.get('Number') 回傳的是 list（如 ['Plur']），不是字串 'Plur'")
    print("     所以 == 'Sing' 永遠是 False，所有名詞都被判成 SINGULAR")
    print("  2. 講義的判斷邏輯寫反了：把 'Sing' 標成 PLURAL、else 標成 SINGULAR")
    print()
    print("lemma 方法正確，因為 spaCy 知道 geese 的 lemma 是 goose，")
    print("goose ≠ geese，所以判斷為 PLURAL。")
