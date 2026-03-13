# Week 3 快速參考卡

## 🎯 四題答案速查

### Q1: NLTK 如何分句？

```
NLTK用Punkt Tokenizer（統計模型），不是單純用句點切。
能識別縮寫（Dr., Mr.）、小數（98.6）、引號等，用機率判斷真正的句子邊界。

結果：11 句
```

---

### Q2: 速度比較

```
1. An important difference is the time

2. The time that spacy uses is [50-300倍] times longer than nltk

原因：spaCy 載入完整模型，NLTK 只做斷句
```

---

### Q3: 斷詞差異

```
1. The word list is longer when using spaCy

2. spaCy keeps the newlines

3. spaCy splits words with a hyphen

NLTK: 230 詞
spaCy: 251 詞
```

---

### Q4: 中文歧義句

```
句子：他知道這件事不要緊

理解 1：他知道 | 這件事不要緊
→ 他知道這件事情不重要

理解 2：他知道這件事 | 不要緊
→ 他知道這件事情，沒關係

歧義原因：斷句位置 + 缺少標點
```

---

## 🔑 關鍵概念

### Punkt Tokenizer

- **類型：** 統計學習模型（非硬性規則）
- **功能：** 判斷真正的句子邊界
- **特點：** 識別縮寫、小數、引號

### NLTK vs spaCy

| 項目 | NLTK | spaCy |
|------|------|-------|
| 速度 | 快 | 慢（50-300倍） |
| 詞數 | 少（230） | 多（251） |
| 換行 | 過濾 | 保留 |
| 連字號 | 保留 | 分割 |

### 中文 NLP 挑戰

- ❌ 無空格分詞
- ❌ 歧義句多
- ❌ 需上下文
- ✅ 需專門工具（jieba）

---

## 💻 快速執行

```bash
cd week3

# Q1
python3 question_01_dividing_sentences/demo.py

# Q2
python3 question_02_compare_speed/compare_speed.py

# Q3
python3 question_03_tokens_comparison/compare_tokens.py

# Q4
python3 question_04_chinese_ambiguity/analyze_ambiguity.py
```

---

## 📝 TronClass 回答模板

### Q1
```
[貼執行結果的 list]
11

NLTK用Punkt Tokenizer（統計模型），不是單純用句點切。
能識別縮寫（Dr., Mr.）、小數（98.6）、引號等。
```

### Q2
```
1. the time
2. spacy uses is [你的倍數] times longer than nltk

NLTK: [你的時間] s
spaCy: [你的時間] s
```

### Q3
```
longer
hyphen

NLTK: 230 words
spaCy: 251 words
```

### Q4
```
理解 1：他知道 | 這件事不要緊（這事不重要）
理解 2：他知道這件事 | 不要緊（沒關係）

歧義原因：斷句位置 + 缺少標點
```

---

## 🎓 考前重點

1. **Punkt 原理：** 統計學習，非硬性規則
2. **速度差異原因：** spaCy 功能多 vs NLTK 單一功能
3. **斷詞差異：** 換行 + 連字號處理不同
4. **中文挑戰：** 無空格、多歧義、需上下文

---

**存檔日期：** 2026.03.13
**複習時直接看這份！** 🚀
