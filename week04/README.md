# Week 4: Lexical and Grammar

輔仁大學 王冠云老師 - 自然語言處理課程
2026.03.20

---

## 課程主題

本週學習如何利用語法資訊（grammatical information）從文本中提取結構化資料。

---

## 資料夾結構

```
week04/
├── README.md
├── 04_LexcialandGrammar_20260320.pdf   # 課程講義
├── question_01_identify_plurals/       # Q1: 單複數判斷
│   ├── identify_plurals.py
│   └── answer.md
├── question_02_compare_spacy_llm/      # Q2: SpaCy vs LLM 比較
│   └── answer.md
├── question_03_dependency_chinese/     # Q3: 中文依存句法分析
│   └── answer.md
└── question_04_hard_or_easy/           # Q4: 哪組句子較難分析
    └── answer.md
```

---

## 重點整理

### 1. Counting Nouns — 單複數判斷

用 spaCy 判斷名詞是 singular 還是 plural，有兩種方法：

- **Lemma 比較法**：比較 `token.lemma_` 和 `token.text`，不同就是複數
- **Morph 屬性法**：用 `token.morph.get("Number")` 直接取得單複數標記

不規則名詞（goose/geese, child/children, foot/feet）小模型容易判斷錯誤，需要用 `en_core_web_lg` 或 LLM 處理。

轉換單複數可用 **TextBlob** 的 `pluralize()` / `singularize()`。

### 2. Dependency Parse — 依存句法分析

依存句法揭示句子中詞與詞之間的語法關係，ROOT 通常是動詞。

spaCy 提供的相關屬性：

| 屬性 | 說明 |
|------|------|
| `token.dep_` | 語法功能標籤（nsubj, dobj, ROOT...） |
| `token.ancestors` | 該詞依賴的上層詞（往 ROOT 方向） |
| `token.children` | 依賴該詞的下層詞 |
| `token.lefts` / `token.rights` | 左側/右側的 children |
| `token.subtree` | 該詞的完整子樹（含所有後代） |

中文也可以用 `zh_core_web_sm` 做依存分析，但效果與英文有差異。

### 3. Noun Chunks — 名詞組提取

名詞組（noun phrase）= 名詞 + 修飾它的詞。

```python
for chunk in doc.noun_chunks:
    print(chunk.text)
```

注意：**中文不支援 `noun_chunks`**。

### 4. Subject / Object Extraction — 主詞受詞提取

透過 `dep_` 標籤提取：
- 包含 `"subj"` → 主詞
- 包含 `"dobj"` → 直接受詞

用 `token.subtree` 可以取得完整的主詞/受詞片語，而不只是單一詞。

### 5. Pattern Matching — 語法模式匹配

用 spaCy 的 `Matcher` 以 POS 標籤定義語法模式來搜尋文本：

```python
from spacy.matcher import Matcher
matcher = Matcher(model.vocab)
patterns = [
    [{"POS": "VERB"}],                              # paints
    [{"POS": "AUX"}, {"POS": "VERB"}],               # was observing
    [{"POS": "AUX"}, {"POS": "ADJ"}],                # were late
    [{"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}] # were staring at
]
matcher.add("Verb", patterns)
```

可用 POS、正則、其他屬性組合出各種模式。

---

## 隨堂測驗

### Q1: Can the model correctly identify plurals?

用 spaCy small model 的 morph 和 lemma 兩種方法判斷 "Three geese crossed the road" 的名詞單複數。morph 方法因程式碼 bug 判斷錯誤（No），lemma 方法正確（Yes）。

### Q2: Compare SpaCy and LLM

用 spaCy large model 和 LLM 辨識不規則複數。兩者都能正確辨識，但 SpaCy 速度遠快於 LLM（本地運算 vs API 呼叫）。

### Q3: Dependency Parsing with Chinese

對「我很少聽到他提起她的其他名字。」做中文依存句法分析，輸出 dependencies、ancestors、children、subtree 四種結果。

### Q4: Hard or Easy?

比較兩組中文句子的依存分析難度。Group B（口語短句）較難，因為常省略主詞、受詞或動詞，導致 parser 難以建立正確的依存關係。

---

## 使用的套件

```bash
uv pip install spacy textblob
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg   # 處理不規則名詞
python -m spacy download zh_core_web_sm   # 中文分析
```
