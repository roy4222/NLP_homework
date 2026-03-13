#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question 01: NLTK Sentence Splitting Demo
"""

import nltk
import ssl

# 處理 SSL 問題
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
    print("下載 NLTK punkt...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def read_text_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# 讀取文本
sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1_short.txt")

# 使用 NLTK Punkt Tokenizer
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
sentences_nltk = tokenizer.tokenize(sherlock_holmes_part_of_text)

# 輸出結果
print(sentences_nltk)
print(len(sentences_nltk))

print("\n" + "=" * 70)
print("NLTK用Punkt Tokenizer（統計模型），不是單純用句點切。")
print("能識別縮寫（Dr., Mr.）、小數（98.6）、引號等，用機率判斷真正的句子邊界。")
print("=" * 70)
