# Week 7: Classification

輔仁大學 王冠云老師 - 自然語言處理課程
2026.04.10

---

## 課程主題

本週學習四種文字分類方法（rule-based / K-Means / SVM / LLM），並用 Rotten Tomatoes + BBC News 英文資料做示範。作業 A3 要求用中文資料重跑 K-Means 並加分題比較 LLM。

---

## 資料夾結構

```
week07/
├── README.md
├── 07_Classification_20260410.pdf       # 課程講義
├── A3_classification.ipynb              # 作業 A3: K-Means 中文新聞分類 + LLM bonus
├── A3_classification.pdf                # 作業 A3 匯出 PDF（繳交用）
├── build_notebook.py                    # 產生 notebook 的 Python script（可重現）
└── exploration/
    └── inspect_wikinews.py              # 資料探索腳本
```

---

## 四種分類方法重點整理

### 1. Rule-based — 關鍵字比對

每個類別整理出獨有字庫（`positive_filtered = set(positive) - intersection`），比對時看哪個類別命中字多。

- 簡單快速、零訓練
- 漏詞就掛（同義詞、口語、新詞）
- 缺乏泛化能力

### 2. K-Means — 非監督式分群

用 TF-IDF 把文件轉向量，`KMeans(n_clusters=k, n_init=10)` 分群。

- 完全不需要標籤資料
- 分幾群要自己決定（elbow / silhouette）
- 每群要看 top words 人工命名
- 沒有 ground truth 就**不能算 accuracy**，只能看一致性

### 3. SVM — 監督式分類

`SentenceTransformer` 轉 embedding → `SVC(kernel='rbf')` 分類。

- 準確率高（簡報 BBC 範例 ~95%）
- 需要標註資料
- 可算 confusion matrix、precision/recall

### 4. LLM — Zero-shot 分類

把文字加提示丟給 LLM（簡報用 Mistral），要求輸出類別。

- 不需要訓練
- 中文 zero-shot 語意理解最強
- 成本高（API 費、速度慢）

---

## 作業 A3: Chinese News K-Means + LLM Bonus

### 作業內容

把 BBC 英文示範改成**繁體中文新聞 K-Means 分類**，使用 `erhwenkuo/wikinews-zhtw` 資料集（9,827 筆，無 label）。加分題：用 Groq + Qwen3-32B 做同樣分類並比較兩者的**同意率**（不是 accuracy — 因為沒 gold label）。

### 做了什麼

1. **資料探索**：9,827 筆 wikinews，發現**沒有 label 欄位**，平均 450 字，長度從 13 到 35k 字不等
2. **前處理**：過濾 < 100 字的 stub、jieba 斷詞、自建 50 詞中文停用詞
3. **TF-IDF**：`ngram_range=(1,2)`, `min_df=5`, `max_df=0.9`, `max_features=20000`
4. **K-Means sweep**（作業沒明寫但我延伸加的）：試 `n_clusters = 5, 7, 10`，比較 silhouette + 每群 top 20 詞
5. **人工命名 7 個 cluster**：體育賽事 / 香港政治 / 疫情醫療 / 香港天氣 / 台灣新聞 / 雜項國際 / 國際政治
6. **PCA 2D 視覺化**（非必做）+ 每群抽 2 篇文章人工驗證
7. **Bonus (Qwen3-32B)**：抽 50 篇做兩個實驗
   - A. Closed-set：給 LLM 我們命名的 7 類，算 **consistency**（不是 accuracy — 沒 ground truth）
   - B. Open-coded：讓 LLM 自由命名主題，看跟 K-Means 群怎麼對應

### 核心發現

| 指標 | 數字 | 觀察 |
|---|---|---|
| 資料規模 | 9,827 → ~9,100（過濾後）| ~7% 是 stub 要丟 |
| 最佳 n_clusters | 7 | silhouette 幾乎沒差（0.014~0.017），要看 readability |
| Silhouette (k=7) | 0.015 | **低但正常** — 短文稀疏向量都這樣 |
| LLM Consistency (placeholder 名) | **8%** | LLM 看不懂 `cluster_0/1/...`，亂選 |
| LLM Consistency (真實名) | **74%** | 有意義的標籤才有用；不是 accuracy（沒 gold label） |
| LLM 獨立命名的主題數 | 22 | 比 K-Means 的 7 類細很多 |

### 三個最大的方法論收穫

1. **Silhouette 在短文本根本分不出 k 的優劣**（三個 k 值都 0.014–0.017），永遠要人工看 top words
2. **Label 命名不是裝飾**：給 LLM `cluster_0` 跟給它「體育賽事」結果差 10 倍（8% vs 74%）— 命名是設計決策
3. **K-Means 一定會有一個 catch-all 垃圾桶**：體育 / 天氣 / 疫情這些詞彙獨特的類很乾淨，但「雜項國際」那群 4,196 篇是所有曖昧文章的終站 — 加 k 只是換地方丟

### 實際應用

1. 新聞網站的 **自動分類 / 版面配置**（Yahoo 新聞、LINE TODAY）
2. **內容推薦冷啟動**（沒用戶歷史時用同群文章當 baseline）
3. **輿情監測 / 新主題偵測**（突發事件會形成自己的群）
4. **檔案索引 / 建立資料驅動分類系統**（法條、判決書、論文都適用）
5. **重複內容偵測**（同群 + top tokens 近似 → 可能是重複報導）

---

## 使用的套件

```bash
uv pip install datasets scikit-learn jieba numpy pandas matplotlib openai tqdm
# Bonus 需要 Groq API key：
export GROQ_API_KEY=gsk_xxx
```

## 重現實驗

```bash
cd /Users/lubaiyu/Desktop/nlp_homework && source .venv/bin/activate
python week07/build_notebook.py  # 重建 notebook
GROQ_API_KEY=$YOUR_KEY jupyter nbconvert --to notebook --execute \
    week07/A3_classification.ipynb --output A3_classification.ipynb
jupyter nbconvert --to webpdf --allow-chromium-download week07/A3_classification.ipynb
```
