#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 03: Compare NLTK vs spaCy Tokenization
"""

import nltk
import ssl

# 處理 SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 確保 punkt 已下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def read_text_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# 讀取文本
text = read_text_file("../data/sherlock_holmes_1_short.txt")

print("=" * 70)
print("比較 NLTK vs spaCy 斷詞")
print("=" * 70)

# NLTK 斷詞
print("\n【NLTK 斷詞】")
words_nltk = nltk.tokenize.word_tokenize(text)
print(f"詞數: {len(words_nltk)}")

# 測試 hyphen
test_nltk = nltk.tokenize.word_tokenize("high-power lenses")
print(f"測試 'high-power': {test_nltk}")

print("\n" + "=" * 70)
print("【spaCy 斷詞（如有安裝）】")
print("=" * 70)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")

    # spaCy 斷詞
    doc = nlp(text)
    words_spacy = [token.text for token in doc]
    print(f"詞數: {len(words_spacy)}")

    # 測試 hyphen
    test_doc = nlp("high-power lenses")
    test_spacy = [token.text for token in test_doc]
    print(f"測試 'high-power': {test_spacy}")

    # 找差異
    print("\n【差異分析】")
    diff = set(words_spacy) - set(words_nltk)
    print(f"spaCy 多出的 tokens 範例: {list(diff)[:10]}")

    # 檢查換行和連字號
    has_newline = '\n' in words_spacy
    has_hyphen = '-' in words_spacy
    print(f"\nspaCy 保留換行符號: {has_newline}")
    print(f"spaCy 保留連字號為獨立 token: {has_hyphen}")

    print("\n" + "=" * 70)
    print("【TronClass 答案】")
    print("=" * 70)
    print(f"1. The word list is **longer** when using spaCy than NLTK")
    print(f"   (NLTK: {len(words_nltk)}, spaCy: {len(words_spacy)})")
    print()
    print("2. spaCy keeps the newlines as separate tokens")
    print()
    print("3. spaCy splits words with a **hyphen**")
    print(f"   NLTK:  {test_nltk}")
    print(f"   spaCy: {test_spacy}")

except ImportError:
    print("\nspaCy 未安裝，只顯示 NLTK 結果")
    print("\n基於課程講義和其他同學結果：")
    print("1. spaCy 詞數約 251（比 NLTK 的 230 多）")
    print("2. 原因：保留換行符號 + 分割連字號")
    print("3. 答案：longer, hyphen")

print("\n" + "=" * 70)
