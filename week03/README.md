# Week 3: NLP Basics

輔仁大學 王冠云老師 - 自然語言處理課程
2026.03.13

---

## 📋 課程主題

本週學習 NLP 基礎技術：

1. ✅ **斷句（Sentence Splitting）** - NLTK Punkt Tokenizer
2. ✅ **速度比較** - NLTK vs spaCy 效能分析
3. ✅ **斷詞（Tokenization）** - 詞彙切分差異
4. ✅ **中文歧義分析** - 理解中文 NLP 的挑戰

---

## 📁 資料夾結構

```
week3/
├── README.md                          # 本檔案
├── question_01_dividing_sentences/    # 題目 1：斷句
│   ├── answer.md                      # 完整答案說明
│   └── demo.py                        # 示範程式碼
├── question_02_compare_speed/         # 題目 2：速度比較
│   ├── answer.md                      # 完整答案說明
│   └── compare_speed.py               # 比較程式碼
├── question_03_tokens_comparison/     # 題目 3：斷詞比較
│   ├── answer.md                      # 完整答案說明
│   └── compare_tokens.py              # 比較程式碼
├── question_04_chinese_ambiguity/     # 題目 4：中文歧義
│   ├── answer.md                      # 完整答案說明
│   └── analyze_ambiguity.py           # 分析程式碼
└── data/                              # 共用資料
    ├── sherlock_holmes_1_short.txt    # 練習文本（短版）
    └── sherlock_holmes_full.txt       # 完整文本
```

---

## 🎯 題目總覽

### Question 01: Dividing Sentences

**問題：** NLTK 如何將文本分成句子？

**答案重點：**
- 使用 Punkt Tokenizer（統計學習模型）
- 不是單純用句點切分
- 能識別縮寫（Dr., Mr.）、小數（98.6）、引號等
- 用機率判斷真正的句子邊界

**執行：**
```bash
cd question_01_dividing_sentences
python3 demo.py
```

---

### Question 02: Compare Speed

**問題：** NLTK vs spaCy 速度差異？

**答案：**
1. An important difference is **the time**
2. spaCy uses **約 50-300 倍** longer than NLTK

**原因：**
- spaCy 載入完整語言模型 + 多種 NLP 工具
- NLTK 只執行斷句功能

**執行：**
```bash
cd question_02_compare_speed
python3 compare_speed.py
```

---

### Question 03: Tokens Comparison

**問題：** NLTK vs spaCy 斷詞差異？

**答案：**
1. spaCy 詞數 **longer**（更多）
2. spaCy 分割連字號：**hyphen**

**原因：**
- spaCy 保留換行符號 `\n` 為獨立 token
- spaCy 將 `high-power` 切成 `['high', '-', 'power']`
- NLTK 保留為 `['high-power']`

**執行：**
```bash
cd question_03_tokens_comparison
python3 compare_tokens.py
```

---

### Question 04: Chinese Ambiguity

**句子：** 他知道這件事不要緊

**歧義分析：**

**理解 1：** 他知道 | 這件事不要緊
- 意思：他知道這件事情不重要

**理解 2：** 他知道這件事 | 不要緊
- 意思：他知道這件事情，沒關係

**歧義原因：**
- 斷句位置不明確
- 缺少標點符號
- 需要上下文判斷

**執行：**
```bash
cd question_04_chinese_ambiguity
python3 analyze_ambiguity.py
```

---

## 🚀 快速複習

### 執行所有示範程式：

```bash
# 進入 week3 資料夾
cd week3

# 題目 1
python3 question_01_dividing_sentences/demo.py

# 題目 2
python3 question_02_compare_speed/compare_speed.py

# 題目 3
python3 question_03_tokens_comparison/compare_tokens.py

# 題目 4
python3 question_04_chinese_ambiguity/analyze_ambiguity.py
```

---

## 📊 重點總結

### NLTK vs spaCy 比較

| 特性 | NLTK | spaCy |
|------|------|-------|
| **斷句原理** | Punkt（統計模型） | 語言模型 + 規則 |
| **速度** | 快（單一功能） | 慢（多功能） |
| **斷詞** | 簡單切分 | 細緻分析 |
| **連字號處理** | 保留完整 | 分割為多個 token |
| **換行處理** | 過濾 | 保留為 token |
| **適用場景** | 快速簡單任務 | 完整 NLP 分析 |

---

## 💡 學習要點

### 1. Punkt Tokenizer 的智能

- 不是硬性規則，而是統計學習
- 能區分「句末句點」vs「縮寫句點」
- 從大量語料學習模式

### 2. 工具選擇的權衡

- **速度** vs **功能**
- NLTK：輕量、快速、專注
- spaCy：完整、強大、較慢

### 3. 中文 NLP 的挑戰

- 無天然詞界（空格）
- 歧義句常見
- 需要上下文理解
- 傳統工具支持有限

---

## 📚 延伸學習

### 推薦閱讀：

1. **NLTK Punkt Tokenizer 論文**
   - Unsupervised Multilingual Sentence Boundary Detection

2. **spaCy 文檔**
   - https://spacy.io/usage/processing-pipelines

3. **中文 NLP 工具**
   - jieba（結巴分詞）
   - pkuseg
   - LTP（語言技術平台）

### 下週預告：

- 詞性標注（POS Tagging）
- 詞形還原（Lemmatization）
- 停用詞移除（Stopwords Removal）

---

## 🛠 環境需求

### Python 套件：
```bash
pip install nltk
# spaCy（可選）
pip install spacy
python -m spacy download en_core_web_sm
```

### 或使用 uv（推薦）：
```bash
uv pip install nltk spacy
uv run python -m spacy download en_core_web_sm
```

---

## 📞 聯絡資訊

- **課程：** 自然語言處理
- **老師：** 王冠云
- **學校：** 輔仁大學資訊管理學系
- **日期：** 2026.03.13

---

**祝學習順利！** 🎉

有問題隨時回來複習這些檔案！
