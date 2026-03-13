# NLP Homework

輔仁大學 資管三 — 自然語言處理（114 學年度第 2 學期）

授課教師：王冠云 ｜ Friday D2,D3,D4 / BS440

## 課程進度與作業

| 週次 | 主題 | 資料夾 | 狀態 |
|:----:|------|--------|:----:|
| 1 | Course Orientation / Environment Setup | — | — |
| 2 | Basic Linguistics | — | — |
| 3 | NLP Basics | [`week3/`](./week3/) | ✅ |
| 4 | Lexical and Grammar | `week04/` | |
| 5 | Semantics (N-gram, TF-IDF, Embeddings) | `week05/` | |
| 6 | Implementation & Spring Vacation | `week06/` | |
| 7 | Classification | `week07/` | |
| 8 | Mid-term Report | `week08/` | |
| 9 | Information Extraction (Regex, NER) | `week09/` | |
| 10 | Implementation & Labor Day | `week10/` | |
| 11 | Topic Modeling (LDA, K-Means) | `week11/` | |
| 12 | Visualizing Text Data | `week12/` | |
| 13 | Transformers and Applications | `week13/` | |
| 14 | Natural Language Understanding | `week14/` | |
| 15 | Generative AI and LLM | `week15/` | |
| 16 | Final Report | `week16/` | |

## 已完成的作業

### Week 3: NLP Basics (2026.03.13)
- TronClass 課堂練習：NLTK 斷句、速度比較、斷詞差異、中文歧義分析
- Assignment 1：中文斷詞與詞性標註（jieba vs spaCy 比較）

### Assignment 1: Tokenization & POS Tagging
- 檔案：[`A1_tokenization_pos.ipynb`](./A1_tokenization_pos.ipynb) / [`A1_tokenization_pos.pdf`](./A1_tokenization_pos.pdf)
- 繳交期限：2026/03/27 08:30

## 環境

- Python >= 3.9
- 套件管理：[uv](https://github.com/astral-sh/uv)（推薦）

```bash
uv venv
uv pip install nltk jieba spacy
```

## 教科書

Antić, Z., & Chakravarty, S. (2024). *Python Natural Language Processing Cookbook*, 2nd Edition. Packt.
