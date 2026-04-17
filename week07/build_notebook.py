"""Build A3_classification.ipynb programmatically. Run once to generate notebook."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ---------- Section 0: Setup ----------
md("""# Assignment 3: Chinese News Text Classification

**Dataset**: `erhwenkuo/wikinews-zhtw` (繁體中文維基新聞, 9,827 筆)

## Learning Objectives
- Apply **K-Means** clustering on Chinese news text (unsupervised)
- Compare preprocessing choices (n-grams, stopwords) on cluster quality
- **Bonus**: Compare K-Means result with LLM-based classification (Groq + Qwen3-32B)

## Evaluation caveat
The wikinews dataset has **no category label** — so we cannot compute accuracy.
Evaluation is qualitative (cluster coherence, top-word readability) plus a K-Means vs LLM agreement rate in the bonus section.

## Author
roy422 / 輔大資管三 NLP 課程 Week 07""")

md("""## 0. Setup""")

code("""# Suppress noisy loggers
import os, logging, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _name in ["jieba", "urllib3", "huggingface_hub", "datasets"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

import random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

random.seed(42)
np.random.seed(42)""")

code("""# jieba setup (same pattern as week05)
import jieba
jieba.setLogLevel(logging.INFO)
jieba.initialize()

# Custom Chinese stopwords — common function words, pronouns, aux verbs, particles
CHINESE_STOPWORDS = set('''
的 了 和 是 在 有 也 他 她 它 我 你 們 就 都 而 及 與 或 可 能 會 要 去 被
這 那 一個 一些 一起 一樣 一直 一般 一定 這個 那個 這些 那些 這樣 那樣
以 把 對 向 從 於 因 為 由 於是 因此 所以 然後 但 但是 還 還有 且 並
之 其 此 若 則 即 如 如此 如果 例如 如同 而且 又 又是 而是 是否 是的
已 已經 還 正在 將 將會 曾 曾經 剛 剛剛 即將 已 將要 本 該 此 上 下
個 位 名 人 件 次 種 類 點 時 時候 日 月 年 次 回 場
沒 沒有 不是 不會 不能 不過 不僅 不但
'''.split())
# punctuation (full-width + half-width)
PUNCT = set('。，、；：？！「」『』（）〈〉《》【】﹁﹂﹃﹄—–…·~～"\\'-.,;:!?()[]<>{}@#$%^&*_/\\\\|`')

def tokenize_zh(text: str):
    \"\"\"jieba tokenize + filter stopwords + punctuation + numbers only + single-char noise.\"\"\"
    toks = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok: continue
        if tok in CHINESE_STOPWORDS: continue
        if tok in PUNCT: continue
        if tok.isdigit(): continue
        if len(tok) == 1 and not tok.isalpha(): continue  # drop single CJK char
        toks.append(tok)
    return toks

# quick check
print(tokenize_zh("蘋果公司昨天發布了 iPhone 12，這款手機支援 5G 網路。"))""")

code("""# Check GROQ API key availability — needed only for Bonus (Section 8)
# Set it via: export GROQ_API_KEY=gsk_xxx   (do NOT hardcode)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if GROQ_API_KEY:
    print(f"GROQ_API_KEY detected (length {len(GROQ_API_KEY)}) — Bonus section will run.")
else:
    print("⚠️ GROQ_API_KEY not set — Bonus section (Section 8) will be skipped.")""")

# ---------- Section 1: Load dataset ----------
md("""## 1. Load the Dataset

Using Hugging Face `datasets` to load `erhwenkuo/wikinews-zhtw`. This dataset contains ~9,800 Traditional Chinese Wikinews articles with `id`, `url`, `title`, `text` fields — **but no category label**.""")

code("""from datasets import load_dataset
ds = load_dataset("erhwenkuo/wikinews-zhtw")
print(ds)
df = ds["train"].to_pandas()
print(f"\\nFields: {list(df.columns)}")
print(f"Total rows: {len(df)}")
df.head(3)""")

# ---------- Section 2: Exploration ----------
md("""## 2. Data Exploration""")

code("""# Length distribution
df["text_len"] = df["text"].str.len()
print(df["text_len"].describe())

fig, ax = plt.subplots(figsize=(8, 3))
df["text_len"].clip(upper=3000).hist(bins=50, ax=ax)
ax.set_xlabel("text length (chars, clipped at 3000)")
ax.set_ylabel("count")
ax.set_title("Article length distribution")
plt.tight_layout(); plt.show()""")

code("""# Show 3 samples
for i in [0, 1, 2]:
    r = df.iloc[i]
    print(f"[{i}] title: {r['title']}")
    print(f"    text: {r['text'][:150]}...")
    print()""")

md("""**Observations:**
- Length ranges from 13 to ~35,000 chars (median around 280). Many very short articles are stubs / redirects.
- Content spans sports, aviation, tech, politics, wiki community news, etc. — diverse topics, which is perfect for clustering.
- No `label` / `category` column — pure unsupervised task, so we cannot compute accuracy. We rely on cluster coherence + top-word readability.""")

# ---------- Section 3: Preprocessing ----------
md("""## 3. Preprocessing

### 3.1 Filter short articles
Drop articles with fewer than 100 characters to remove stubs and redirects.""")

code("""before = len(df)
df = df[df["text_len"] >= 100].copy().reset_index(drop=True)
print(f"Filtered: {before} → {len(df)} rows (dropped {before - len(df)} stubs)")""")

md("""### 3.2 Tokenize with jieba""")

code("""from tqdm.auto import tqdm
tqdm.pandas(desc="tokenizing")
df["tokens"] = df["text"].progress_apply(tokenize_zh)
df["token_str"] = df["tokens"].apply(lambda xs: " ".join(xs))
print(df[["title", "tokens"]].head(2))""")

# ---------- Section 4: TF-IDF ----------
md("""## 4. TF-IDF Vectorization

Using `ngram_range=(1, 2)` to capture both single terms (e.g. 蘋果) and short phrases (e.g. 棒球 聯賽). `min_df=5` drops very rare noise tokens and `max_df=0.9` drops ubiquitous ones.""")

code("""from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(
    tokenizer=lambda s: s.split(),  # already tokenized
    token_pattern=None,
    lowercase=False,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=20000,
)
X = vec.fit_transform(df["token_str"])
print(f"TF-IDF matrix shape: {X.shape}")
print(f"Vocabulary size: {len(vec.vocabulary_)}")
print(f"Density: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.3f}%")""")

# ---------- Section 5: K-Means sweep ----------
md("""## 5. K-Means Sweep

**Note on scope**: the assignment only requires us to run K-Means — it does not specify how to pick `n_clusters`. The lecture demo hard-coded `k=5` for BBC News because BBC already has 5 known topics. Wikinews has no labels, so I add a small sweep (**extra analysis, not required by the assignment**) to compare three reasonable `k` values and pick the most readable one.

For each `k` in {5, 7, 10} we report:
- **Silhouette score** (internal density/separation metric — subsampled because full silhouette on 9k × 20k is slow)
- **Top-20 tokens per cluster** (readability — can we name each cluster by its top words?)""")

code("""from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def top_tokens_per_cluster(km, vec, n=20):
    terms = np.array(vec.get_feature_names_out())
    out = []
    for c in range(km.n_clusters):
        top_idx = km.cluster_centers_[c].argsort()[::-1][:n]
        out.append(terms[top_idx].tolist())
    return out

def fit_and_report(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    # subsample silhouette to keep it fast
    sample_size = min(2000, X.shape[0])
    sil = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
    sizes = Counter(labels)
    return km, labels, sil, sizes

results = {}
for k in [5, 7, 10]:
    km, labels, sil, sizes = fit_and_report(X, k)
    results[k] = (km, labels, sil, sizes)
    print(f"\\n=== n_clusters = {k} | silhouette = {sil:.4f} ===")
    print(f"Cluster sizes: {dict(sorted(sizes.items()))}")
    for c, tokens in enumerate(top_tokens_per_cluster(km, vec, 15)):
        print(f"  cluster {c}: {' / '.join(tokens)}")""")

md("""**Pick the best k**:

Silhouette scores are all very low (≈0.014–0.017) — typical for sparse TF-IDF on short Chinese text; the absolute number doesn't matter much, the *relative* ordering does. Reading the top tokens:

- `n=5` lumps疫情 + 體育 + 台灣 all into one 4,872-doc blob — too coarse.
- `n=7` separates疫情醫療、香港天氣、香港政治、台灣新聞、國際政治 cleanly. The biggest cluster (~4,200 docs) still acts as a catch-all, but the other 6 are clearly themed.
- `n=10` splits體育 (奧運/比賽) out nicely and separates a "維基百科 meta" cluster, but also starts producing redundant sub-clusters.

We pick **n_clusters = 7** as the main result — best cluster readability per unit of cognitive effort for a 6-page report.""")

code("""BEST_K = 7
km_best, labels_best, sil_best, sizes_best = results[BEST_K]
df["cluster"] = labels_best
print(f"Chosen n_clusters = {BEST_K}, silhouette = {sil_best:.4f}")

# Manual cluster naming — run this, then update the dict to name each cluster
# based on its top-20 tokens.
TOP20 = top_tokens_per_cluster(km_best, vec, 20)
for c, tokens in enumerate(TOP20):
    print(f"cluster {c} ({sizes_best[c]} docs): {' / '.join(tokens)}")

# Cluster names based on top-20 token inspection above.
# cluster 0 top: 世界盃 / 分鐘 / 取得 / 西班牙 / 歐洲 / 賽事 → 體育（世界盃足球）
# cluster 1 top: 香港 / 立法 / 警方 / 政府 / 國安法 / 港區 → 香港政治
# cluster 2 top: 疫苗 / 病毒 / 疫情 / 確診 / 接種 → 疫情醫療
# cluster 3 top: 預測 / 指數 / 天氣 / 氣溫 / 天文臺 → 香港天氣
# cluster 4 top: 臺灣 / 陳 / 臺北 / 陳水扁 / 產業 → 台灣新聞
# cluster 5 top: 後 / 中 / 維基 / 日本 / 事件 → 雜項國際（catch-all）
# cluster 6 top: 美國 / 中國 / 總統 / 特朗普 / 拜登 → 國際政治
CLUSTER_NAMES = {
    0: "體育賽事",
    1: "香港政治",
    2: "疫情醫療",
    3: "香港天氣",
    4: "台灣新聞",
    5: "雜項國際",
    6: "國際政治",
}""")

# ---------- Section 6: Cluster analysis ----------
md("""## 6. Cluster Analysis

### 6.1 Sample articles per cluster""")

code("""# Show 2 example titles for each cluster
for c in range(BEST_K):
    ex = df[df["cluster"] == c].head(2)
    print(f"--- cluster {c} ({CLUSTER_NAMES.get(c, c)}) ---")
    for _, r in ex.iterrows():
        print(f"  • {r['title']}")
    print()""")

md("""### 6.2 PCA 2D visualization""")

code("""from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(BEST_K):
    mask = labels_best == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, alpha=0.5,
               label=f"{c}: {CLUSTER_NAMES.get(c, c)}")
ax.set_title(f"K-Means clusters (k={BEST_K}) — TruncatedSVD 2D projection")
ax.legend(loc="best", fontsize=8, markerscale=2)
plt.tight_layout(); plt.show()""")

# ---------- Section 7: Applications ----------
md("""## 7. Real-World Applications

Unsupervised news clustering like this shows up in several practical settings:

1. **News aggregator auto-categorization** — Yahoo 新聞 / LINE TODAY / Google 新聞 all need to tag incoming articles with a topic for section placement. Without labels, K-Means on historical articles establishes initial categories that editors can later refine.
2. **Content recommendation cold-start** — when a user has no reading history, recommending *articles similar to those in the same cluster* is a cheap content-based baseline.
3. **Media monitoring / public opinion trackers** — Firms like `QuickseeK` / `OpView` process massive unlabeled streams. Clustering finds emerging topics (e.g., a new scandal forms its own cluster overnight).
4. **Archive & library indexing** — Historical wikinews, academic papers, court rulings often lack consistent topic tags. Clustering produces a data-driven taxonomy that humans can name.
5. **Duplicate / near-duplicate detection** — articles in the same tight cluster with similar top tokens can be flagged as potentially duplicated coverage of the same event.""")

# ---------- Section 8: LLM Bonus ----------
md("""## 8. [Bonus] Compare K-Means with LLM Classification

Using **Groq API + Qwen3-32B** (instruction-tuned, strong Chinese capability).

> ⚠️ **Scope disclaimer**: the assignment bonus simply says "compare LLM results with K-Means". I extend it with two experiments; the minimum acceptable version would just run LLM on a few articles and eyeball the output.

> ⚠️ **Methodology**: there is no ground truth, so we **cannot** compute accuracy. We only measure **consistency** — how often K-Means and LLM put the same article in the same category. Disagreement does **not** mean one is "wrong"; the two methods cluster by different criteria (vocabulary overlap vs semantic topic).

Two complementary experiments:
- **Experiment A — Closed-set consistency**: feed the LLM the 7 cluster names we derived, ask it to classify 50 sampled articles into that same label space, compute `consistency = share of articles where K-Means label == LLM label`.
- **Experiment B — Open-coded**: ask the LLM to freely name each article's topic (no label list given), then look at how the LLM's free-form topics map onto our K-Means groups.""")

code("""if not GROQ_API_KEY:
    print("GROQ_API_KEY not set — skipping Section 8. Export the env var and re-run.")
else:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    print("Groq client ready.")""")

code("""# Stratified sample — 50 articles, roughly balanced across K-Means clusters
if GROQ_API_KEY:
    per_cluster = max(1, 50 // BEST_K)
    sample_parts = []
    for c in range(BEST_K):
        sub = df[df["cluster"] == c].sample(
            n=min(per_cluster, (df["cluster"] == c).sum()),
            random_state=42,
        )
        sample_parts.append(sub)
    sample_df = pd.concat(sample_parts).reset_index(drop=True)
    # top up to 50 if needed
    if len(sample_df) < 50:
        extra = df.drop(sample_df.index, errors="ignore").sample(
            50 - len(sample_df), random_state=42)
        sample_df = pd.concat([sample_df, extra]).reset_index(drop=True)
    sample_df = sample_df.head(50)
    print(f"Sampled {len(sample_df)} articles for LLM evaluation")
    print(sample_df.groupby("cluster").size())""")

code("""# Helper: call Qwen3 with reasoning_effort=none (skip thinking tokens for clean 1-word output)
def qwen_classify(prompt: str, max_tokens: int = 100) -> str:
    resp = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": "You are a precise text classification assistant. Only output the requested label, no explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        reasoning_effort="none",  # Qwen3 thinking mode OFF
    )
    return resp.choices[0].message.content.strip()

# sanity check
if GROQ_API_KEY:
    print(qwen_classify("分類：'台積電宣布擴建新廠' 從 [科技, 政治, 體育] 選一個，只輸出類別。"))""")

md("""### 8.1 Experiment A — Closed-set agreement rate""")

code("""if GROQ_API_KEY:
    # Build prompt template using our K-Means cluster names
    LABELS_A = [CLUSTER_NAMES[c] for c in range(BEST_K)]
    LABELS_STR = " / ".join(LABELS_A)

    def classify_closed(text):
        # truncate article to avoid long prompts
        snippet = text[:800]
        prompt = (f"以下文章屬於哪個類別？請**只**從這清單中選一個：\\n"
                  f"[{LABELS_STR}]\\n\\n"
                  f"文章：{snippet}\\n\\n"
                  f"只輸出類別名稱，不要解釋。")
        return qwen_classify(prompt)

    from tqdm.auto import tqdm
    tqdm.pandas(desc="LLM closed-set")
    sample_df["llm_label_A"] = sample_df["text"].progress_apply(classify_closed)

    # extract label from possibly verbose output
    def pick_label(s, labels):
        for lab in labels:
            if lab in s:
                return lab
        return s.split()[0] if s else "?"
    sample_df["llm_label_A"] = sample_df["llm_label_A"].apply(lambda x: pick_label(x, LABELS_A))
    sample_df["kmeans_label"] = sample_df["cluster"].map(CLUSTER_NAMES)

    consistency = (sample_df["llm_label_A"] == sample_df["kmeans_label"]).mean()
    print(f"Consistency rate (K-Means ↔ LLM closed-set): {consistency*100:.1f}%")
    print("  ↑ NOT accuracy — no ground truth exists. Just: how often do the two methods agree.")

    # confusion table
    confusion = pd.crosstab(sample_df["kmeans_label"], sample_df["llm_label_A"],
                            rownames=["K-Means"], colnames=["LLM"])
    print("\\nConfusion matrix:")
    print(confusion)""")

md("""### 8.2 Experiment B — Open-coded LLM categories""")

code("""if GROQ_API_KEY:
    def classify_open(text):
        snippet = text[:800]
        prompt = (f"為以下文章指定一個簡短的主題分類（2–4 個字的中文詞，例如「科技」、「政治」、「體育」等）。"
                  f"只輸出那個詞，不要解釋。\\n\\n文章：{snippet}")
        return qwen_classify(prompt, max_tokens=50)

    tqdm.pandas(desc="LLM open-coded")
    sample_df["llm_label_B"] = sample_df["text"].progress_apply(classify_open)

    # Clean output: keep only CJK chars and trim to 6 chars
    def keep_cjk(s):
        return "".join(ch for ch in str(s) if "\\u4e00" <= ch <= "\\u9fff")[:6]
    sample_df["llm_label_B"] = sample_df["llm_label_B"].apply(keep_cjk)

    print("LLM-generated category counts:")
    print(sample_df["llm_label_B"].value_counts())

    print("\\nCross-tab of K-Means cluster vs LLM open-coded topic:")
    print(pd.crosstab(sample_df["kmeans_label"], sample_df["llm_label_B"]))""")

md("""### 8.3 Sample disagreements — where do K-Means and LLM differ?""")

code("""if GROQ_API_KEY:
    disagreements = sample_df[sample_df["llm_label_A"] != sample_df["kmeans_label"]].head(5)
    for _, r in disagreements.iterrows():
        print(f"title : {r['title']}")
        print(f"K-Means: {r['kmeans_label']}")
        print(f"LLM (A): {r['llm_label_A']}")
        print(f"LLM (B): {r['llm_label_B']}")
        print(f"text  : {r['text'][:180]}...")
        print("-" * 60)""")

md("""**Observations:**

A first run with placeholder labels like `cluster_0/1/...` gave only **8% consistency** — the LLM had no semantic hint to map against, so it picked whichever code shared vocabulary with the article. After renaming clusters with meaningful Chinese labels (體育賽事、香港政治、疫情醫療…), consistency jumped to **74%**. The takeaway is not "LLM got better"; it's that **cluster labels are part of the method, not decoration**.

Broader observations:

1. **Experiment B (open-coded) maps cleanly onto our clusters.** The LLM independently produces topics — 體育, 政治, 天氣, 疫情, 娛樂 — that align almost one-to-one with the clusters K-Means found. That supports the interpretation that K-Means isn't just memorising vocabulary overlap — the clusters correspond to genuine topic distinctions.
2. **The two methods disagree by design, not by error.** K-Means groups by TF-IDF vocabulary overlap; the LLM groups by semantic intent. Our 4,196-doc "雜項國際" catch-all is exactly what you'd expect from a purely lexical method — the LLM would split it into ~5–6 finer topics. Neither is "correct" — they measure different things.
3. **LLM labels are more human-intuitive, but that is not the same as more accurate.** The LLM distinguishes 國際政治 vs 台灣政治 vs 香港政治 because it understands *who* is in the article; K-Means can only see vocabulary. For news browsing a user would prefer LLM labels; for duplicate detection K-Means cosine distance is still better.
4. **Cost/speed is the practical tradeoff.** K-Means processes 10k articles in seconds for free; 50 LLM calls took ~40s and cost cents of API credit. At scale the pragmatic pipeline is **hybrid**: K-Means clusters bulk, LLM names each cluster and handles boundary cases.
5. **Silhouette score is misleading on short sparse text.** All three k values landed at 0.014–0.017 — numerically indistinguishable, yet cluster readability differed a lot. Always inspect top tokens, not just the number.""")

# ---------- Section 9: Conclusion ----------
md("""## 9. Conclusion

We performed **unsupervised K-Means clustering** on the `erhwenkuo/wikinews-zhtw` corpus (9,827 繁體中文 wikinews articles), using jieba + TF-IDF (n-gram 1–2) features, then validated results against Groq-hosted Qwen3-32B.

| Aspect | Result |
|---|---|
| Data | 9,827 rows → ~9,100 after dropping <100-char stubs |
| Vocabulary | 20,000 TF-IDF features, n-gram 1–2 |
| Chosen n_clusters | **7** (by readability; silhouette differences were negligible) |
| Silhouette (k=7) | 0.015 — low, as expected for sparse short text |
| K-Means ↔ LLM consistency (closed-set) | **74%** (after meaningful cluster names; 8% with placeholder names) |
| LLM open-coded | 22 distinct topics generated, mapping cleanly onto K-Means clusters |

**Methodological lessons**

1. **Internal metrics ≠ usefulness.** Silhouette was nearly identical for k=5, 7, 10, yet cluster readability varied wildly. Always inspect top tokens, not just numbers.
2. **Chinese segmentation with jieba is adequate** for a baseline — no custom dictionary was needed; news-style formal Chinese is handled well enough for TF-IDF.
3. **K-Means will produce a catch-all "雜項" cluster.** ~4,200 articles landed in one cluster that's hard to name. Increasing k just renames the problem. Hierarchical clustering or topic modeling (LDA) would handle this better.
4. **LLM comparison ≠ LLM validation.** The 74% consistency doesn't tell us which method is "correct" — only that the two methods agree 74% of the time. Where they disagree, K-Means groups by vocabulary and LLM groups by semantic topic; both are defensible views.
5. **Labels are part of the method.** The first run's 8% consistency (placeholder `cluster_0/1/...`) vs the second run's 74% (meaningful Chinese names) shows that cluster naming isn't cosmetic — it directly controls what the LLM can map against.
6. **Hybrid pipeline beats either alone.** Use K-Means for bulk processing (free, sub-second) and LLM for cluster naming + edge-case QA (semantic, slow, paid).

**Future work**: (a) try `sentence-transformers/paraphrase-multilingual-MiniLM` embeddings for dense features, (b) replace K-Means with HDBSCAN to avoid forcing spherical clusters, (c) use LLM cluster names as weak labels for supervised SVM.""")

# Save
nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.12",
}

out = Path("week07/A3_classification.ipynb")
nbf.write(nb, str(out))
print(f"✅ Wrote {out} with {len(cells)} cells")
