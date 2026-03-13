# Question 01: Dividing Sentences

## 📋 題目

Observe the code in the following:

```python
import nltk
nltk.download('punkt')

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
sentences_nltk = tokenizer.tokenize(sherlock_holmes_part_of_text)
print(sentences_nltk)
print(len(sentences_nltk))
```

**問題：Do you know how NLTK divides a text into sentences?**

---

## ✅ TronClass 答案

```
['To Sherlock Holmes she is always _the_ woman.', 'I have seldom heard him\nmention her under any other name.', 'In his eyes she eclipses and\npredominates the whole of her sex.', 'It was not that he felt any emotion\nakin to love for Irene Adler.', 'All emotions, and that one particularly,\nwere abhorrent to his cold, precise but admirably balanced mind.', 'He\nwas, I take it, the most perfect reasoning and observing machine that\nthe world has seen, but as a lover he would have placed himself in a\nfalse position.', 'He never spoke of the softer passions, save with a gibe\nand a sneer.', 'They were admirable things for the observer—excellent for\ndrawing the veil from men's motives and actions.', 'But for the trained\nreasoner to admit such intrusions into his own delicate and finely\nadjusted temperament was to introduce a distracting factor which might\nthrow a doubt upon all his mental results.', 'Grit in a sensitive\ninstrument, or a crack in one of his own high-power lenses, would not\nbe more disturbing than a strong emotion in a nature such as his.', 'And\nyet there was but one woman to him, and that woman was the late Irene\nAdler, of dubious and questionable memory.']
11

NLTK用Punkt Tokenizer（統計模型），不是單純用句點切。
能識別縮寫（Dr., Mr.）、小數（98.6）、引號等，用機率判斷真正的句子邊界。
```

---

## 📖 詳細說明

### NLTK 如何分句？

**使用工具：Punkt Sentence Tokenizer（統計學習模型）**

#### 工作機制：

1. **智能判斷，不是硬性規則**
   - 不是遇到句點就切分
   - 使用機率模型判斷句子邊界

2. **縮寫識別**
   - Dr. Watson → 識別為縮寫（不分割）
   - Mr. Holmes → 識別為縮寫（不分割）
   - woman. I → 句末 + 大寫（分割）

3. **上下文分析**
   - 檢查句點後是否大寫
   - 分析詞彙長度
   - 考慮前後語境

4. **特殊處理**
   - 小數點：98.6 degrees（不分割）
   - 引號："Holmes!"（正確處理）
   - 省略號：...（不誤判）

#### 核心概念：

**Punkt 從大量英文語料統計學習，用機率模型判斷真正的句子邊界，而非使用「遇到句點就切」的硬性規則。**

---

## 🔬 驗證測試

```python
test = "Dr. Watson met Mr. Holmes at 221B Baker St. He smiled."
sentences = tokenizer.tokenize(test)
# 結果：['Dr. Watson met Mr. Holmes at 221B Baker St.', 'He smiled.']
# 正確！不會在 Dr./Mr./St. 處誤切
```

---

## 📚 參考資料

- PDF 講義第 5 頁
- NLTK Punkt Tokenizer: [文檔連結](https://www.nltk.org/api/nltk.tokenize.punkt.html)
