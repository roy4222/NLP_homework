# Backend container for the LexTag prediction API.
# Portable to Render / Railway / Fly.io / any Docker host.
# The TF-IDF+SVM model is rebuilt at build time from the committed data/train.jsonl,
# so the image needs no Hugging Face access at build or run time.
FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn / scipy wheels are already in slim; install python deps.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy only what the API needs to build the model and serve.
COPY api/ ./api/
COPY baselines/ ./baselines/
COPY scripts/ ./scripts/
COPY data/issues.yaml data/train.jsonl data/test.jsonl data/law_articles.json ./data/

# Rebuild the TF-IDF + SVM model from committed training data (offline, deterministic).
RUN python baselines/tfidf_svm.py && test -f data/tfidf_svm.joblib

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Render/Fly/Railway inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "gunicorn 'api.app:create_app()' --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120"]
