"""
Week 05 - Challenge: Using Cohere LLM Embeddings for sentiment classification
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import cohere
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import os
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "YOUR_KEY_HERE")

# ── Load data (same as 02_comparison) ───────────────────────────────────

def load_train_test_dataset_pd():
    train_dataset = load_dataset("rotten_tomatoes", split="train[:15%]+train[-15%:]")
    test_dataset = load_dataset("rotten_tomatoes", split="test[:15%]+test[-15%:]")
    train_df = train_dataset.to_pandas()
    train_df = train_df.sample(frac=1, random_state=42)
    test_df = test_dataset.to_pandas()
    return train_df, test_df

print("Loading datasets...")
train_df, test_df = load_train_test_dataset_pd()
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ── Get Cohere embeddings ───────────────────────────────────────────────

co = cohere.Client(api_key=COHERE_API_KEY)

def get_embeddings_batch(texts, input_type):
    """Get embeddings in batches of 96 (Cohere limit)."""
    all_embeddings = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type=input_type,
        )
        all_embeddings.extend(response.embeddings)
    return all_embeddings

print("\nGetting Cohere embeddings...")
start = time.time()

train_embeddings = get_embeddings_batch(
    train_df["text"].tolist(), input_type="classification")
test_embeddings = get_embeddings_batch(
    test_df["text"].tolist(), input_type="classification")

elapsed = time.time() - start
print(f"Cohere embeddings processing time: {elapsed:.2f} s")

# ── Train & evaluate classifier ─────────────────────────────────────────

X_train = np.array(train_embeddings)
X_test = np.array(test_embeddings)
y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

print(f"Embedding dimension: {X_train.shape[1]}")

clf = LogisticRegression(C=0.1, max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nCohere embed-english-v3.0 Results:")
print(classification_report(y_test, y_pred))
