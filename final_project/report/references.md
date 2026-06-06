# References

## Datasets

- `lianghsun/tw-legal-synthetic-qa`. Hugging Face dataset. Main weak-label source for Chinese legal QA scenarios.
- `lianghsun/tw-processed-law-article`. Hugging Face dataset. Processed Taiwanese law article source used for statute lookup and ontology support.
- `aigrant/taiwan-ly-law-research`. Hugging Face dataset. Optional out-of-domain legal research evaluation source.

## Methods and Libraries

- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
- scikit-learn documentation. `TfidfVectorizer`, one-vs-rest classification, linear models, and multi-label metrics.
- Jieba Chinese text segmentation documentation.
- Hugging Face Transformers documentation, for the optional Chinese BERT extension.
- Flask documentation, for the project API layer.

## Course Topics Used

- Sentence processing and tokenization.
- N-gram and TF-IDF features.
- Rule-based classification.
- SVM or linear one-vs-rest classification.
- Evaluation with F1 score, hamming loss, and per-class analysis.

## Project Files

- `final_project/README.md`
- `final_project/FINAL_PROJECT_SPEC.md`
- `final_project/DATA_USABILITY.md`
- `final_project/data/issues.yaml`
- `final_project/tests/`
