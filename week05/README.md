# Week 5: Semantics

輔仁大學 王冠云老師 - 自然語言處理課程
2026.03.27

---

## 課程主題

本週學習如何把文字轉成數值向量（text representation），並用不同方法做情感分類與語意相似度比較。

---

## 資料夾結構

```
week05/
├── README.md
├── 05_Semantics_20260327.pdf              # 課程講義
├── 02_comparison.py                       # 討論 02: 五種方法情感分類比較
├── 03_challenge_cohere.py                 # 討論 03: Cohere LLM embeddings 挑戰
├── A2_representation_embeddings.ipynb     # 作業 A2: TF-IDF vs Embeddings（中文）
├── A2_representation_embeddings.pdf       # 作業 A2 匯出 PDF（繳交用）
└── similarity_comparison.png              # Heatmap 視覺化比較圖
```

---

## 重點整理

### 1. POS Vectorizer — 用詞性統計做向量

用 spaCy 統計每段文字中各詞性（VERB, NOUN, ADJ...）的數量，產生 10 維向量。

- 向量 = [文字長度, VERB數, NOUN數, PROPN數, ADJ數, ADV數, AUX數, PRON數, NUM數, PUNCT數]
- 太粗糙，幾乎沒有語意資訊，accuracy 僅 ~0.58

### 2. Bag of Words — 詞袋模型

用 `CountVectorizer` 把文字轉成詞頻向量，每個維度代表一個詞的出現次數。

- `max_df` 控制停用詞：`max_df=0.4` 表示出現在 40% 以上文件中的詞會被忽略
- 不考慮詞序，只管詞頻
- accuracy ~0.73

### 3. N-gram Model — 考慮詞組合

用 `CountVectorizer(ngram_range=(1,2))` 同時考慮 unigram 和 bigram。

- 詞彙量比 unigram 大很多（因為多了雙詞組合）
- 但 accuracy 不一定更好（~0.74），因為高維度可能引入噪音

### 4. TF-IDF — 加入詞的重要性權重

`TfidfVectorizer` 在詞頻基礎上加入 IDF 權重，讓罕見但有意義的詞獲得更高分數。

- TF = 詞在文件中的出現次數 / 文件總詞數
- IDF = 總文件數 / 包含該詞的文件數
- TF-IDF = TF × IDF
- 也可以用 character n-gram（`analyzer='char_wb'`）
- accuracy ~0.75，本週古典方法中最佳

### 5. Word Embeddings — 詞向量

用 gensim 載入預訓練的 Google Word2Vec 模型（300 維），或自己用 Rotten Tomatoes 語料訓練。

- 每個詞對應一個稠密向量，語意相近的詞在向量空間中距離也近
- 句子向量 = 所有詞向量的平均值（centroid）
- 自訓練模型因語料太小，效果反而很差（~0.50-0.54）
- 有趣功能：`most_similar()`、`doesnt_match()`

### 6. LLM Embeddings — 大型語言模型的嵌入

用 LLM API（如 Cohere、Mistral）取得高品質文字嵌入。

- Cohere `embed-english-v3.0`：1024 維，accuracy **0.89**
- 效果遠超古典方法，但需要 API 費用和網路請求時間

---

## 課堂討論

### 02_Comparison — 五種方法情感分類比較

在 Rotten Tomatoes 電影評論上比較五種 text representation 搭配 LogisticRegression 的分類效果：

| Method | Accuracy |
|--------|----------|
| TF-IDF | **0.75** |
| N-gram (Bigram) | 0.74 |
| Bag-of-Words | 0.73 |
| POS Vectorizer | 0.58 |
| Word Embeddings | 0.50 |

TF-IDF 在古典方法中表現最好，因為它能降低常見詞的權重、提升有區辨力的詞的重要性。

### 03_Challenge — Cohere LLM Embeddings

| Method | Accuracy | Dimension | Time |
|--------|----------|-----------|------|
| Cohere embed-english-v3.0 | **0.89** | 1024 | 18.27s |

LLM embeddings 大幅勝出，因為預訓練模型在海量語料上學到了更豐富的語意表示。

---

## 作業 A2: Text Representation — TF-IDF & Embeddings（中文）

### 作業內容

用中文資料比較 TF-IDF 和 Sentence Embeddings 兩種文字表示法的 cosine similarity 差異。

### 做了什麼

1. **擴充語料**：從 4 句擴展到 8 句中文，涵蓋半導體、NLP、教育等主題
2. **TF-IDF**：用 jieba 斷詞 + `TfidfVectorizer`，印出 top 5 關鍵詞與相似度矩陣
3. **Sentence Embeddings**：用 `paraphrase-multilingual-MiniLM-L12-v2`（384 維）算相似度矩陣
4. **視覺化**：兩張 heatmap 並排比較
5. **關鍵配對分析**：挑 6 組文件對，比較兩種方法的分數差異
6. **Discussion**：分析為什麼 embeddings 更好、TF-IDF 的局限與優勢

### 核心發現

| Document Pair | TF-IDF | Embedding |
|---|---|---|
| 半導體 vs 台積電（同主題不同詞） | 0.143 | **0.671** |
| NLP App vs 文本分類（相關主題） | 0.174 | **0.561** |
| LLM 教育 vs 教育（都談教育） | 0.157 | **0.381** |

結論：**Sentence embeddings 在語意相似度任務上明顯優於 TF-IDF**，因為它能理解「用不同詞講同一件事」。

---

## 使用的套件

```bash
uv pip install datasets scikit-learn spacy gensim numpy pandas cohere sentence-transformers jieba matplotlib
python -m spacy download en_core_web_sm
```
