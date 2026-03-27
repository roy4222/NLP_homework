"""
Week 05 - Semantics: 02_Comparison
Compare POS vectorizer, Bag-of-Words, N-gram, TF-IDF, Word Embeddings
on Rotten Tomatoes sentiment classification.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ── Helper functions (from slides) ──────────────────────────────────────

class POS_vectorizer:
    def __init__(self, spacy_model):
        self.model = spacy_model

    def vectorize(self, input_text):
        doc = self.model(input_text)
        vector = []
        vector.append(len(doc))
        pos = {"VERB":0, "NOUN":0, "PROPN":0, "ADJ":0, "ADV":0,
               "AUX":0, "PRON":0, "NUM":0, "PUNCT":0}
        for token in doc:
            if token.pos_ in pos.keys():
                pos[token.pos_] += 1
        vector_values = list(pos.values())
        vector = vector + vector_values
        return vector


def load_train_test_dataset_pd():
    train_dataset = load_dataset("rotten_tomatoes", split="train[:15%]+train[-15%:]")
    test_dataset = load_dataset("rotten_tomatoes", split="test[:15%]+test[-15%:]")
    train_df = train_dataset.to_pandas()
    train_df = train_df.sample(frac=1, random_state=42)
    test_df = test_dataset.to_pandas()
    return train_df, test_df


def create_train_test_data(train_df, test_df, vectorize_fn):
    train_df["vector"] = train_df["text"].apply(lambda x: vectorize_fn(x))
    test_df["vector"] = test_df["text"].apply(lambda x: vectorize_fn(x))
    X_train = np.stack(train_df["vector"].values, axis=0)
    X_test = np.stack(test_df["vector"].values, axis=0)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train):
    clf = LogisticRegression(C=0.1, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def test_classifier(test_df, clf):
    test_df["prediction"] = test_df["vector"].apply(lambda x: clf.predict([x])[0])
    report = classification_report(test_df["label"], test_df["prediction"])
    return report


# ── Load data ───────────────────────────────────────────────────────────

print("Loading datasets...")
train_df, test_df = load_train_test_dataset_pd()
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

results = {}

# ── 1. POS Vectorizer ───────────────────────────────────────────────────

print("\n" + "="*60)
print("1. POS Vectorizer")
print("="*60)

nlp = spacy.load("en_core_web_sm")
pos_vec = POS_vectorizer(nlp)

train_df_copy, test_df_copy = train_df.copy(), test_df.copy()
X_train, X_test, y_train, y_test = create_train_test_data(
    train_df_copy, test_df_copy, pos_vec.vectorize)
clf = train_classifier(X_train, y_train)
report = test_classifier(test_df_copy, clf)
print(report)
results["POS Vectorizer"] = float(report.split("accuracy")[1].split()[0])

# ── 2. Bag-of-Words ────────────────────────────────────────────────────

print("="*60)
print("2. Bag-of-Words (CountVectorizer, max_df=300)")
print("="*60)

bow_vectorizer = CountVectorizer(max_df=300)
bow_vectorizer.fit(train_df["text"])
vectorize_bow = lambda x: bow_vectorizer.transform([x]).toarray()[0]

train_df_copy, test_df_copy = train_df.copy(), test_df.copy()
X_train, X_test, y_train, y_test = create_train_test_data(
    train_df_copy, test_df_copy, vectorize_bow)
clf = train_classifier(X_train, y_train)
report = test_classifier(test_df_copy, clf)
print(report)
results["Bag-of-Words"] = float(report.split("accuracy")[1].split()[0])

# ── 3. N-gram Model (Bigram) ───────────────────────────────────────────

print("="*60)
print("3. N-gram Model (Bigram, ngram_range=(1,2), max_df=300)")
print("="*60)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=300)
bigram_vectorizer.fit(train_df["text"])
vectorize_bigram = lambda x: bigram_vectorizer.transform([x]).toarray()[0]

train_df_copy, test_df_copy = train_df.copy(), test_df.copy()
X_train, X_test, y_train, y_test = create_train_test_data(
    train_df_copy, test_df_copy, vectorize_bigram)
clf = train_classifier(X_train, y_train)
report = test_classifier(test_df_copy, clf)
print(report)
results["N-gram (Bigram)"] = float(report.split("accuracy")[1].split()[0])

# ── 4. TF-IDF ──────────────────────────────────────────────────────────

print("="*60)
print("4. TF-IDF (max_df=300)")
print("="*60)

tfidf_vectorizer = TfidfVectorizer(max_df=300)
tfidf_vectorizer.fit(train_df["text"])
vectorize_tfidf = lambda x: tfidf_vectorizer.transform([x]).toarray()[0]

train_df_copy, test_df_copy = train_df.copy(), test_df.copy()
X_train, X_test, y_train, y_test = create_train_test_data(
    train_df_copy, test_df_copy, vectorize_tfidf)
clf = train_classifier(X_train, y_train)
report = test_classifier(test_df_copy, clf)
print(report)
results["TF-IDF"] = float(report.split("accuracy")[1].split()[0])

# ── 5. Word Embeddings (Word2Vec via gensim) ───────────────────────────

print("="*60)
print("5. Word Embeddings (self-trained Word2Vec)")
print("="*60)

import gensim
from gensim.models import Word2Vec
from gensim import utils

# Train Word2Vec on the full rotten_tomatoes training set
full_train = load_dataset("rotten_tomatoes", split="train")

class RottenTomatoesCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
    def __iter__(self):
        for review in self.sentences:
            yield utils.simple_preprocess(
                gensim.parsing.preprocessing.remove_stopwords(review))

corpus = RottenTomatoesCorpus(full_train["text"])
w2v_model = Word2Vec(sentences=corpus, vector_size=100, window=5,
                     min_count=1, workers=4)
w2v_model.train(corpus_iterable=corpus,
                total_examples=w2v_model.corpus_count, epochs=100)


def get_word_vectors(sentence, model):
    word_vectors = []
    for word in sentence:
        try:
            word_vector = model.wv[word.lower()]
            word_vectors.append(word_vector)
        except KeyError:
            continue
    return word_vectors


def get_sentence_vector(word_vectors):
    if len(word_vectors) == 0:
        return np.zeros(100)
    matrix = np.array(word_vectors)
    centroid = np.mean(matrix[:, :], axis=0)
    return centroid


vectorize_w2v = lambda x: get_sentence_vector(get_word_vectors(x, w2v_model))

train_df_copy, test_df_copy = train_df.copy(), test_df.copy()
X_train, X_test, y_train, y_test = create_train_test_data(
    train_df_copy, test_df_copy, vectorize_w2v)
clf = train_classifier(X_train, y_train)
report = test_classifier(test_df_copy, clf)
print(report)
results["Word Embeddings"] = float(report.split("accuracy")[1].split()[0])

# ── Summary ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<20s}: {acc:.2f}")

best = max(results, key=results.get)
print(f"\n>> {best} has the best accuracy ({results[best]:.2f}).")
