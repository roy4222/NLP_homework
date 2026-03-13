# NLP_homework

輔仁大學 王冠云老師 - 自然語言處理課程作業

## 📚 課程內容

### Week 3: NLP Basics (2026.03.13)

本週學習 NLP 基礎技術，包含：

1. **斷句（Sentence Splitting）** - NLTK Punkt Tokenizer
2. **速度比較** - NLTK vs spaCy 效能分析
3. **斷詞（Tokenization）** - 詞彙切分差異
4. **中文歧義分析** - 理解中文 NLP 的挑戰

📂 [查看 Week 3 完整內容](./week3/)

---

## 🚀 快速開始

```bash
# 進入 week3 資料夾
cd week3

# 執行各題示範程式
python3 question_01_dividing_sentences/demo.py
python3 question_02_compare_speed/compare_speed.py
python3 question_03_tokens_comparison/compare_tokens.py
python3 question_04_chinese_ambiguity/analyze_ambiguity.py
```

---

## 📖 學習資源

- [Week 3 詳細說明](./week3/README.md)
- [Week 3 快速參考卡](./week3/QUICK_REFERENCE.md)
- [課程講義 PDF](./03_NLP_Basics_20260313.pdf)

---

## 🛠 環境需求

```bash
# 安裝 NLTK
pip install nltk

# 安裝 spaCy（可選）
pip install spacy
python -m spacy download en_core_web_sm
```

或使用 uv（推薦）：
```bash
uv pip install nltk spacy
uv run python -m spacy download en_core_web_sm
```

---

**學校：** 輔仁大學資訊管理學系
**老師：** 王冠云
**日期：** 2026.03.13
