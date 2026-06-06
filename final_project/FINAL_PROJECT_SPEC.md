# 期末專題設計文件 v4

**題目（中文）**：中文法律問題前置分流 — 從生活情境預測法律議題
**題目（英文）**：Chinese Legal Issue Triage — Multi-label Prediction from Everyday Legal Narratives
**課程**：輔大資管三 自然語言處理（王冠云老師）
**文件日期**：2026-06-06（v4：定位為 legal issue triage classifier）
**作者**：roy422

---

## 0. v4 定位

本專題不是法律 AI chatbot，也不是 RAG 法律問答系統，而是 **legal issue triage classifier**：

> 使用者用白話描述法律情境 → 系統預測可能涉及的法律議題 → 作為後續查資料、找律師、補問問題的前置分流。

市場上已有 Lawbot AI、Lawsnote、LawChat、EasyLaw、LawAI 法詢等產品主打法律問答、RAG 搜判決、契約審閱、書狀生成、律師媒合等完整服務。期末專題若直接做「法律 AI 諮詢」會變成低配版競品，缺乏可評估性與可重現資料基礎。

因此 v4 聚焦在競品也需要的前置 NLP 子任務：**中文法律問題理解與多標籤議題分類**。它直接對應 guideline 的 **Text Classification**，可用 micro/macro F1、hamming loss、per-class F1 做清楚評估。

原 v2 設計是「勞資糾紛 RAG + 工具計算 + 前後端」，在實際探索資料後發現：

- `tw-legal-synthetic-qa` 真正含法條引用的資料只有 **41.5% (4,001 筆)**，**勞資相關僅 1.0% (101 筆)** → 勞資 RAG 死路
- 該資料集本質是「情境 → 法律分析」對，天生適合做**分類任務**，不是 RAG
- v2 範圍過大（前端 + 後端 + 工具 + RAG），不符合 guideline「深度 > 廣度」原則

**v4 聚焦單一明確任務**：**多標籤法律議題分流**，直接對應 guideline §2.1 的 **Text Classification** 主題。

---

## 1. 問題 & 使用者

### 1.1 問題
一般人遇到法律問題時，常常**不知道自己的情況屬於哪類法律議題**，因此：
- 不知道要查哪部法
- 不知道要找哪類專業（民事律師？刑事律師？勞工局？）
- 在法律科普網站找不到對的切入點
- 直接問 chatbot 時，問題描述常常太散，容易得到泛泛回答

### 1.2 目標使用者
- **主要**：遇到突發法律問題的一般大眾，想先搞清楚「這是什麼問題」
- **次要**：法律科普平台的初步分流（案件路由）、法扶諮詢專線前置分類、律所 intake 表單、法學院學生自我檢核

### 1.3 使用者故事
> 使用者：「我被告妨害名譽 但是 threads 上的人是冒名的 怎麼辦」
> 系統：`[妨害名譽, 冒名/身分盜用, 個資侵害]`
> 並附上：可能相關條文 `刑法 §310`、`民法 §195`、`個資法 §41`，標註為「查詢線索」，不生成法律建議。

---

## 2. 課程主題對應

Guideline §2.1 列出的 topics 中，本專案精確對應：

> **Text classification (rule-based, SVM, K-Means, spaCy, LLM-based methods)**

方法比較涵蓋課程教過的三種典型路徑：
1. **Rule-based**：關鍵字/別名字典比對
2. **傳統 ML**：TF-IDF + Linear SVM / Logistic Regression (One-vs-Rest)
3. **Deep Learning**：Chinese BERT fine-tune

---

## 3. 資料集

### 3.1 主資料：`lianghsun/tw-legal-synthetic-qa`
- **實際規模**：train 7,704 + test 1,927 = **9,631 筆**
- **格式**：`{messages: [{role: user, content: 情境}, {role: assistant, content: 法律分析}]}`
- **資料探索發現**（見 `final_project/DATA_USABILITY.md`）：
  - **32.8% (3,158 筆)** 有可用法條引用（去除 regex 雜訊後的淨數字）
  - 67.2% 無明確引用 / 只是泛論 → 丟棄
  - 主要領域分布 **刑法佔 77%**（2,434 筆），其他如民法 911、刑訴 598、民訴 280、道交 213
  - 手動抽樣 10 筆驗證 label 抽取品質 ≈ 85–90% 正確
- **授權**：Apache 2.0
- **角色**：主任務資料。從 assistant 法律分析中抽取法條引用，作為 weak labels / distant supervision，再映射成法律議題。

### 3.2 外部 OOD 測試資料：`aigrant/taiwan-ly-law-research`
- **規模**：2,910 筆，其中 1,521 筆有顯式 `related_laws` 標註
- **可用為 OOD 測試**：591 筆（至少含 1 個主資料 label space 內的法規）
- **特色**：立法院議題研析報告，**文風跟主資料不同**（正式報告 vs 故事），適合測跨文風泛化
- **注意**：平均長 2,731 字、52% 超過 2,000 字，BERT 需截斷 1,500 字
- **授權**：Apache 2.0

### 3.3 輔助資料：`lianghsun/tw-processed-law-article`
- **規模**：230,974 rows，欄位包含 `text`, `name`, `level`, `abandon_note`, `modified_date`, `api_updated_date`
- **用途**：議題 ontology 對應的法條原文查詢、法規名稱正規化、法條別名核對、Demo 顯示條文原文
- **不用於**：模型訓練主資料、RAG corpus、生成式回答
- **授權**：CC BY 4.0

### 3.4 不使用（附理由）
- `tw-processed-law-ctx` (11.5k 筆)：整部法規合併長文本，適合 pretrain / lookup，不適合本次分類訓練
- `tw-legal-nlp` (171 筆)：太小，適合參考 NER / QA / 條號轉換任務設計，不適合主訓練資料
- `tw-legal-benchmark-v1` (209 題)：4 選 1 選擇題，不是 story→law 映射
- `aa0101181514/tw-legal-rag`：是連接遠端 TLR endpoint 的 retrieval CLI，不包含開源判決庫、embedding 或向量索引；可借 citation normalization / bundle schema 思路，不作為本專題架構

### 3.5 資料擴充預案（見 IMPLEMENTATION_PLAN.md Phase 1.5）
若 Task 5 完成後訓練量不足或刑法偏重過高（>70%），有四條擴充路徑（optional）：

1. **申請 gated datasets**（10 min）：`tw-legal-qa-3M` (5.4k) + `tw-legal-qa-chat` (527)，同主資料格式，潛在翻倍訓練資料
2. **aigrant 切入訓練**：把 591 OOD 筆切 300 進訓練（aigrant 分布幾乎沒刑法、強力補平衡），291 筆仍留 OOD test
3. **LLM 合成稀有類別**：對 <30 筆議題用 Cerebras Qwen 生成變體補到 50 筆
4. **合併極稀有類別**：<5 筆議題合併成「其他X」bucket

四條可組合或獨立執行。成本：10 分鐘申請 + 半天合成，預期增 1,500–2,000 筆、平衡改善。

---

## 4. 任務定義

- **Input**：中文法律情境敘述（自由文字，50–500 字典型）
- **Output**：一組法律議題標籤（multi-label，每筆平均 1–3 個）
- **標籤空間 MVP**：**15–25 個高頻實體法律議題**（先做刑法 + 民法主軸，跑通後再擴到 30–50）
- **任務類型**：**Multi-label classification**

### 議題粒度決策
- 不用「法規名」（刑法/民法…）：標籤太粗，用戶輸出「這是刑法」沒實用價值
- 不用「條號」（刑法§310…）：標籤長尾嚴重（4,278 個 pair 只出現 1 次），不適合 classification
- 採用「**法律議題**」中間層：例如 妨害名譽、酒駕、詐欺、不當解僱 …，兼顧實用性與訓練可行性
- 第一版排除刑事訴訟法 / 民事訴訟法程序標籤：這些多是「案件進入法院後怎麼處理」（如簡易判決、不受理、撤回告訴），與本專題「一般人法律問題前置分流」目標不同。

### MVP 候選 label set

第一版以高頻刑法 / 民法條文合併成議題：

1. 酒駕 / 公共危險
2. 竊盜
3. 加重竊盜
4. 詐欺
5. 傷害
6. 過失傷害
7. 過失致死
8. 肇事逃逸
9. 毀損
10. 偽造文書
11. 妨害名譽
12. 恐嚇
13. 侵權行為 / 損害賠償
14. 離婚
15. 不當得利
16. 契約責任 / 債務不履行
17. 買賣瑕疵
18. 繼承
19. 夫妻財產
20. 所有物返還 / 物權請求

---

## 5. NLP Pipeline — 八步驟

### Step 0 — 法條引用正規化
**做什麼**：處理全半形、`台/臺`、法規別名、空白、`第185條之3` 等格式，將引用統一成 `(法規名, 條號, 之幾)`。

**產出**：`final_project/scripts/citation_normalizer.py` + unit tests

**工作量**：~0.5 天

### Step 1 — 議題 Ontology 建構
**做什麼**：從 `tw-legal-synthetic-qa` 抽取高頻 (法規, 條號) pair → 查該條文內容 → 人工對應到議題名 → 寫白話別名與關鍵詞

**產出**：`final_project/data/issues.yaml`
```yaml
- id: defamation
  name: 妨害名譽
  aliases: [誹謗, 公然侮辱, 說我壞話, 肉搜, 罵人]
  laws:
    - [刑法, 310]
    - [刑法, 313]
    - [民法, 195]
  keywords: [名譽, 誹謗, 侮辱, 人格權]
```

**工作量**：~1 天

### Step 2 — 自動打標籤
**做什麼**：
1. 對每筆 row 跑 citation normalizer 從 assistant content 抽 (法規, 條號)
2. 查 ontology 映射到議題 ID
3. 一筆可有多 label
4. 標記 label quality：`usable`, `needs_review`, `discard`
5. 丟掉：無標籤 / assistant 表示「資訊不足」 / user 句過短

**產出**：`final_project/data/labeled.jsonl`
```json
{"story": "...", "labels": ["defamation", "privacy_violation"], "raw_citations": [["刑法", "310"]]}
```

**工作量**：~0.5 天

### Step 3 — 資料清洗 & 切分
**做什麼**：去重、長度過濾、stratified split (70/15/15)、類別分布統計
**產出**：`data/{train,val,test}.jsonl` + `data/stats.md`（類別分布直方圖、類別不平衡比率）
**工作量**：~0.5 天

### Step 4 — Rule-based Baseline
**做什麼**：議題的 aliases + keywords 比對，命中數 > threshold → 該 label = 1
**產出**：`baselines/rule.py` + metrics
**預期**：F1 差（尤其 macro），但作為後續模型的對照組
**工作量**：~0.5 天

### Step 5 — TF-IDF + Linear Classifier
**做什麼**：
- jieba 斷詞（加自訂法律詞典：刑法、妨害名譽、資遣費 …）
- TF-IDF (char 1-2 gram + word 1-gram) vectorize
- One-vs-Rest Logistic Regression 與 Linear SVM 比較
- Threshold tuning via validation set

**產出**：`baselines/tfidf_svm.py` + metrics + feature importance 分析（每議題的 top features）
**工作量**：~1 天

### Step 6 — BERT Fine-tune
**做什麼**：
- 模型：`hfl/chinese-macbert-base`（備案 `ckiplab/bert-base-chinese`）
- HuggingFace Trainer，BCEWithLogitsLoss + sigmoid head
- Hyperparameters：batch 16, lr 2e-5, 5 epochs, early stopping on val macro-F1

**產出**：`models/bert_finetune.py` + checkpoint + metrics
**工作量**：~2 天（含 debug + 訓練時間）

### Step 7 — 評估 + 錯誤分析 + Demo
**做什麼**：
- 三法比較表：micro/macro F1、hamming loss、per-class F1
- Confusion heatmap、長尾議題表現分析
- 挑 5–10 個錯誤案例做 case study（FP/FN 原因）
- Demo notebook：輸入自由文字 → 三模型並排對照 + 預測議題 + 建議法條

**產出**：`eval/metrics.py`, `eval/error_analysis.ipynb`, `demo.ipynb`
**工作量**：~1.5 天

---

## 6. 評估方法

### 6.1 量化指標
| 指標 | 用途 |
|---|---|
| Micro-F1 | 整體表現，大類樣本主導 |
| **Macro-F1** | 類別不平衡下的真實表現（**主要指標**）|
| Hamming Loss | 多標籤錯誤率 |
| Per-class F1 | 看長尾議題的個別表現 |

### 6.2 質化
- Error analysis：5–10 筆錯誤案例，分類 FP/FN 原因（同義詞缺失？議題混淆？訓練樣本不足？）
- Confusion heatmap：找出高混淆議題對（例：詐欺 vs 背信）

### 6.3 Demo 驗證
- 找 3 位沒學過法律的同學試用，記錄「模型預測結果是否符合直覺」
- 錄 2 分鐘 demo 影片（口頭報告備案，防當機）

---

## 7. 範圍（MoSCoW）

### MUST HAVE
- [ ] 議題 ontology MVP (15–25 類，法條來源可追溯)
- [ ] 訓練集 ≥2,000 筆 labeled data（基於 3,158 可用總量 70% split 估計）
- [ ] Rule-based / TF-IDF+SVM 端對端實作，BERT 視時間納入
- [ ] 量化比較表（in-domain）+ 錯誤分析
- [ ] Demo notebook（現場 demo 用）
- [ ] Presentation slides + 8–10 分鐘英文口頭簡報

### SHOULD HAVE
- [ ] 使用者研究（3 位同學試用）
- [ ] 議題 → 條文原文顯示（掛 `tw-processed-law-article` 當 lookup 表）
- [ ] BERT fine-tune
- [ ] OOD aigrant 測試
- [ ] 錄 demo 影片備案

### WON'T HAVE（明確不做）
- ❌ 前端網站
- ❌ LLM 生成式回覆
- ❌ 法律建議生成 / AI 律師
- ❌ RAG 系統 / 工具計算器
- ❌ 判決搜尋 / 判決摘要
- ❌ 書狀生成 / 契約審閱
- ❌ 從頭 pretrain 或 domain adaptation pretrain
- ❌ 多輪對話
- ❌ K-Means（議題已有監督標籤，unsupervised 無必要；報告簡略提及不做的原因即可）

---

## 8. 風險

| 風險 | 機率 | 影響 | 對策 |
|---|---|---|---|
| 議題 ontology label noise | 中 | 中 | Step 1 手動檢查，Step 2 抽樣 100 筆人工驗證打標 |
| 長尾議題 F1 差 | 高 | 低 | **當賣點分析**，而非硬解；誠實展示類別不平衡 |
| BERT fine-tune 不收斂 | 低 | 中 | 備案：降 lr、freeze 底層、改用 embedding + LR |
| 資料偏刑事領域 | 高 | 中 | Limitations 誠實寫，避免過度宣稱「普適於法律諮詢」 |
| 期末時程壓縮 | 中 | 中 | 保留 1 週 buffer，SHOULD HAVE 可犧牲 |

---

## 9. 技術棧

| 層級 | 選型 | 理由 |
|---|---|---|
| Python | 3.12 + uv | 既有環境 |
| 斷詞 | jieba + 自訂法律詞典 | week 3 用過、輕量 |
| Feature | scikit-learn TfidfVectorizer | 標準工具 |
| 傳統 ML | sklearn LogisticRegression / LinearSVC (OvR) | multi-label 直接支援 |
| Transformer | HuggingFace transformers + PyTorch | 標準 |
| Base model | `hfl/chinese-macbert-base` | 繁中表現佳、size 適中 |
| 評估 | sklearn metrics | 標準 |
| Demo | Jupyter Notebook | 口頭報告現場跑 |
| 報告 | Markdown → PDF (pandoc) | 既有流程 |

---

## 10. 目錄結構

```
final_project/
├── README.md
├── pyproject.toml
├── data_exploration/
│   ├── inspect_datasets.py        ✅ 已完成
│   └── inspect_v2.py              ✅ 已完成
├── data/
│   ├── issues.yaml                ← Step 1
│   ├── labeled.jsonl              ← Step 2
│   ├── train.jsonl                ← Step 3
│   ├── val.jsonl
│   ├── test.jsonl
│   └── stats.md
├── scripts/
│   ├── build_ontology.py          ← Step 1
│   ├── auto_label.py              ← Step 2
│   └── split_data.py              ← Step 3
├── baselines/
│   ├── rule.py                    ← Step 4
│   └── tfidf_svm.py               ← Step 5
├── models/
│   └── bert_finetune.py           ← Step 6
├── eval/
│   ├── metrics.py
│   └── error_analysis.ipynb
├── demo.ipynb                     ← 口頭報告現場跑
└── report/
    ├── report.md
    ├── report.pdf
    └── slides.pdf
```

---

## 11. 時程

| Phase | Steps | 天數 |
|---|---|---|
| Phase 1：資料準備 | Step 1–3 | 2 天 |
| Phase 2：Baselines | Step 4–5 | 1.5 天 |
| Phase 3：BERT | Step 6 | 2 天 |
| Phase 4：評估 & 分析 | Step 7 | 1.5 天 |
| Phase 5：報告 & 演練 | — | 1–2 天 |
| **Total** | | **8–9 天（可並行壓至 5–6 天）** |

---

## 12. Guideline 交付物對照（vs §2.4）

| Guideline 要求 | 本專案對應 |
|---|---|
| Code implementation | `final_project/` 整個資料夾（scripts + baselines + models + notebooks）|
| Presentation slides | `report/slides.pdf` |
| Oral presentation 8–10 分鐘 | `report/slides.pdf` + `demo.ipynb` 現場跑 |
| Dataset description | `report.pdf` §3（含 exploration 結果）|

---

## 13. 評分對照（vs §3）

| 項目 | 權重 | 本專案主張 |
|---|---|---|
| Technical Complexity 20% | 議題 ontology 建構 + 自動打標 pipeline + 多標籤分類 + 三法比較 + BERT fine-tune + 類別不平衡處理 + 錯誤分析 |
| Practical Applicability 30% | 明確使用者（遇法律問題的一般人）+ 具體應用（法律 AI intake triage / 法扶路由 / 律所 intake / 法學自檢）|
| Implementation Quality 30% | Pipeline 分層清楚（scripts/baselines/models 獨立）+ 量化評估完整（3 指標）+ 錯誤分析誠實 + code 可重現 |
| Oral Presentation 20% | Demo 現場輸入自由文字看三模型對照 + 使用者故事（「threads 冒名」）+ 誠實展示 failure case |

---

## 14. 預期最終效果（Demo 範例）

### 現場 Demo 輸入
```
我被告妨害名譽 但是 threads 上的人是冒名的 怎麼辦
```

### 三模型對照輸出
| 議題 | Rule-based | TF-IDF + SVM | BERT |
|---|---|---|---|
| 妨害名譽 | ✓ (keyword hit) | 0.87 ✓ | 0.94 ✓ |
| 冒名/身分盜用 | ✗ (miss) | 0.41 | 0.78 ✓ |
| 個資侵害 | ✗ | 0.52 ✓ | 0.71 ✓ |

### Demo 結構化輸出
```
模型預測議題：妨害名譽、冒名/身分盜用、個資侵害
可能相關條文（查詢線索，非法律建議）：
  ‣ 刑法 §310 誹謗罪
  ‣ 刑法 §313 妨害信用
  ‣ 個人資料保護法 §41
```

### 預期量化成績（預估，實跑調整）
| Method | Micro-F1 | Macro-F1 | Hamming Loss |
|---|---|---|---|
| Rule-based | 0.42 | 0.28 | 0.15 |
| TF-IDF + SVM | 0.71 | 0.58 | 0.07 |
| BERT-finetune | 0.83 | 0.74 | 0.04 |

---

## 15. 參考資源

### 資料集
- `lianghsun/tw-legal-synthetic-qa`：https://huggingface.co/datasets/lianghsun/tw-legal-synthetic-qa
- `lianghsun/tw-processed-law-article`：https://huggingface.co/datasets/lianghsun/tw-processed-law-article
- `lianghsun/tw-processed-law-ctx`：https://huggingface.co/datasets/lianghsun/tw-processed-law-ctx
- `lianghsun/tw-legal-nlp`：https://huggingface.co/datasets/lianghsun/tw-legal-nlp
- `lianghsun/tw-legal-benchmark-v1`：https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1

### 競品 / 相關產品
- Lawbot AI：https://lawbot.tw/landing
- Lawsnote：https://about.lawsnote.com/
- LawChat：https://lawchat.com.tw/
- EasyLaw：https://easy-law.net/
- LawAI 法詢：https://www.lawfavor.com/

### 模型
- `hfl/chinese-macbert-base`：https://huggingface.co/hfl/chinese-macbert-base
- `ckiplab/bert-base-chinese`：https://huggingface.co/ckiplab/bert-base-chinese

### 參考研究
- "A Survey on Multi-label Text Classification" (2022)
- 法學線上資源：全國法規資料庫 https://law.moj.gov.tw/

---

## 16. 下一步

1. Spec 通過 → 用 writing-plans skill 寫出 **逐步可執行的 implementation plan**
2. Plan 通過 → 開始 Step 1（議題 ontology 建構，預計 1 天內產出 `issues.yaml`）
3. 每完成一個 Step 就 commit、更新報告對應章節
