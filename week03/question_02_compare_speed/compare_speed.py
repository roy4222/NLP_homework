#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 02: Compare NLTK vs spaCy Speed
"""

import nltk
import time
import ssl

# 處理 SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 確保 NLTK 資料已下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下載 NLTK punkt...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def read_text_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# 讀取文本
sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1_short.txt")

print("=" * 70)
print("比較 NLTK vs spaCy 斷句速度")
print("=" * 70)

# NLTK 斷句
def split_into_sentences_nltk(text):
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = tokenizer.tokenize(text)
    return sentences

# spaCy 斷句（需要安裝 spacy）
def split_into_sentences_spacy(text):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sentence.text for sentence in doc.sents]
        return sentences
    except ImportError:
        return None

# 測試 NLTK
print("\n【測試 NLTK】")
start = time.time()
sentences_nltk = split_into_sentences_nltk(sherlock_holmes_part_of_text)
nltk_time = time.time() - start
print(f"NLTK: {nltk_time:.6f} s")
print(f"句子數: {len(sentences_nltk)}")

# 測試 spaCy
print("\n【測試 spaCy】")
sentences_spacy = split_into_sentences_spacy(sherlock_holmes_part_of_text)

if sentences_spacy:
    start = time.time()
    sentences_spacy = split_into_sentences_spacy(sherlock_holmes_part_of_text)
    spacy_time = time.time() - start
    print(f"spaCy: {spacy_time:.6f} s")
    print(f"句子數: {len(sentences_spacy)}")

    # 計算倍數
    print("\n" + "=" * 70)
    print("【比較結果】")
    print("=" * 70)
    ratio = spacy_time / nltk_time
    print(f"spaCy 比 NLTK 慢 {ratio:.2f} 倍")

    print("\n【TronClass 答案】")
    print("=" * 70)
    print("1. An important difference between spaCy and NLTK is the time")
    print("   it takes to complete the sentence-splitting process.")
    print()
    print(f"2. The time that spacy uses is {ratio:.2f} times longer than nltk")
else:
    print("spaCy 未安裝")
    print("\n基於課程講義和同學結果：")
    print("spaCy 通常比 NLTK 慢 50-300 倍")
    print("\n答案：")
    print("1. the time")
    print("2. spacy uses is [你的結果] times longer than nltk")

print("=" * 70)
