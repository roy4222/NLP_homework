# CLAUDE.md

## 專案說明

這是輔仁大學資管三「自然語言處理」課程的作業區。
授課教師：王冠云，上課時間：Friday D2,D3,D4 / BS440，114 學年度第 2 學期。

## 資料夾結構

每週作業放在對應的 `weekXX/` 資料夾中，格式如下：

```
nlp_homework/
├── CLAUDE.md
├── README.md
├── .gitignore
├── week03/   ← NLP Basics（斷句、斷詞、POS tagging）
├── week04/   ← Lexical and Grammar
├── week05/   ← Semantics
├── ...
└── 各週課程講義 PDF
```

注意：week01 是環境建置、week02 是語言學導論，這兩週沒有作業，所以資料夾從 week03 開始。
目前 `week3/` 已存在（不帶前導零），後續新週次請統一用 `weekXX/` 格式（如 `week04/`）。

## 課程進度對照表

| 週次 | 主題 | 內容 |
|------|------|------|
| 1 | Course Orientation | Python programming, Environment setup |
| 2 | Basic Linguistics | Introduction of Linguistics, English vs. Chinese |
| 3 | NLP Basics | Sentence splitting, Tokenization, POS tagging |
| 4 | Lexical and Grammar | Lexical Database, Corpus, Nouns/Chunks, Subject/Object extraction |
| 5 | Semantics | N-gram, TF-IDF, Word embeddings, BERT embeddings |
| 6 | Implementation & Spring Vacation | 實作練習 |
| 7 | Classification | Rule-based, K-Means, SVM, spaCy model, LLM model |
| 8 | Mid-term Report | 個人/小組報告 |
| 9 | Information Extraction | Regex, Levenshtein distance, NER |
| 10 | Implementation & Labor Day | 實作練習 |
| 11 | Topic Modeling | LDA, Community detection, K-Means topic modeling |
| 12 | Visualizing Text Data | Visualization, Word clouds |
| 13 | Transformers | Tokenizing datasets, Text classification, Zero-shot, Text generation |
| 14 | NLU | Answering questions, Explainability |
| 15 | Generative AI and LLM | Running LLM, Chatbot, Text-to-SQL |
| 16 | Final Report | |
| 17-18 | Self-Learning | Modern Trends of NLP |

## 開發環境

- Python >= 3.9（目前使用 3.12）
- 使用 `uv` 管理虛擬環境（`.venv/` 已加入 .gitignore）
- 每週作業可能用到不同套件，在各週資料夾內用 `uv pip install` 安裝
- 常用套件：nltk, jieba, spacy, stanza, transformers

## 作業規範

- 繳交格式通常為 **PDF**（Jupyter Notebook 含程式碼與執行結果匯出）
- 作業提交平台：TronClass
- 作業內容需要包含自己的觀察與分析，不能只有程式輸出
- 答案（answer.md）一律用英文撰寫

## 注意事項

- 寫作業時要加入自己的觀察（「我原本以為...但實際上...」），不要只列結果
- 分析要挑重點講，不需要每句都展開
- 比較工具時要誠實指出哪個比較差，不要寫太平衡的模板式比較
- commit 訊息用英文，內容描述清楚改了什麼
