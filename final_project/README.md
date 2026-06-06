# Chinese Legal Issue Triage

This final project builds a Chinese NLP classifier for legal issue triage.
It is not a legal advice chatbot and does not generate legal opinions.

## Task

Input: an everyday Chinese legal scenario.

Output: one or more legal issue labels, confidence scores, and supporting law articles as lookup clues.

## Main Datasets

- `lianghsun/tw-legal-synthetic-qa`: main weak-label source.
- `lianghsun/tw-processed-law-article`: statute lookup and ontology support.

## Methods

1. Citation normalization and weak labeling
2. Rule-based baseline
3. TF-IDF + One-vs-Rest SVM

## Commands

```bash
uv sync --extra dev
uv run pytest -v
uv run python scripts/build_labels.py
uv run python scripts/split_data.py
uv run python baselines/rule.py
uv run python baselines/tfidf_svm.py
uv run python eval/metrics.py
uv run python api/app.py
```

## Disclaimer

This project performs legal issue triage for NLP education. It does not provide legal advice.
