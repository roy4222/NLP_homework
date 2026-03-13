# Question 02: Compare the Sentences

## 📋 題目

Execute the codes and complete the sentence:

* An important difference between spaCy and NLTK is ________ it takes to complete the sentence-splitting process.
  (提示: 此空格的答案在講義中)

* The time that `nltk`/`spacy` uses is _____ times longer than `nltk`/`spacy`
  (請比較兩個套件執行時間)

---

## ✅ TronClass 答案

### 填空 1：
```
An important difference between spaCy and NLTK is the time it takes
to complete the sentence-splitting process.
```

### 填空 2：
```
The time that spacy uses is [約 50-300 倍] times longer than nltk

實測結果會因電腦效能而異，大部分同學的結果：
- 最快差異：約 48 倍
- 最慢差異：約 301 倍
- 常見範圍：50-100 倍
```

---

## 📊 執行結果範例

```
NLTK: 0.0003 s
spaCy: 0.0470 s

spaCy 比 NLTK 慢約 156 倍
```

---

## 💡 原因分析

### 為什麼 spaCy 慢很多？

#### spaCy：
- ✅ 載入完整語言模型
- ✅ 同時執行多種 NLP 任務：
  - 斷句
  - 詞性標注
  - 依存句法分析
  - 命名實體識別
  - 等等...
- ⚠️ 功能強大但較慢

#### NLTK：
- ✅ 只執行斷句功能
- ✅ 輕量級
- ✅ 速度快

### 結論：

**spaCy 是功能完整的 NLP 套件，NLTK 的 Punkt 只專注於斷句。**

如果只需要斷句 → 用 NLTK 更快
如果需要多種 NLP 任務 → spaCy 一次處理更方便

---

## 📚 參考資料

- PDF 講義第 7-9 頁
- 同學實測結果：48-301 倍差異
