# Question 03: Tokens from NLTK and spaCy

## 📋 題目

Execute the code and complete the following sentences:

* You will notice that the length of the word list is __shorter/longer__ when using spaCy than NLTK.
* One of the reasons is that spaCy keeps the newlines, and each newline is a separate token.
* The other difference is that spaCy splits words with a ___???__, such as high-power.

---

## ✅ TronClass 答案

```
You will notice that the length of the word list is longer when using spaCy than NLTK.

One of the reasons is that spaCy keeps the newlines, and each newline is a separate token.

The other difference is that spaCy splits words with a hyphen, such as high-power.

NLTK words count: 230
spaCy words count: 251
```

---

## 📊 詳細分析

### 為什麼 spaCy 詞數更多？

#### 原因 1：保留換行符號

**NLTK：**
```python
# 會過濾掉換行符號 \n
words_nltk = ['To', 'Sherlock', 'Holmes', ...]
```

**spaCy：**
```python
# 保留換行符號作為獨立 token
words_spacy = ['To', '\n', 'Sherlock', '\n', 'Holmes', ...]
```

#### 原因 2：分割連字號

**測試：`"high-power lenses"`**

**NLTK：**
```python
['high-power', 'lenses']  # 2 個 token
```

**spaCy：**
```python
['high', '-', 'power', 'lenses']  # 4 個 token
```

---

## 🔍 實測驗證

```python
import nltk

# NLTK
text = "high-power lenses"
nltk_tokens = nltk.tokenize.word_tokenize(text)
print(nltk_tokens)  # ['high-power', 'lenses']

# spaCy（需安裝）
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print(spacy_tokens)  # ['high', '-', 'power', 'lenses']
```

---

## 💡 差異總結

| 特性 | NLTK | spaCy |
|------|------|-------|
| 換行符號 | 過濾 | 保留為獨立 token |
| 連字號詞 | 保留完整 | 分割為多個 token |
| 詞數 | 較少（230） | 較多（251） |
| 目的 | 簡單分詞 | 細緻語言分析 |

---

## 📚 參考資料

- PDF 講義第 16 頁
- 同學實測結果：NLTK 約 230 詞，spaCy 約 251 詞
