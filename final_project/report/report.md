# Chinese Legal Issue Triage: Multi-label Prediction from Everyday Legal Narratives

## 1. Project Overview

This final project builds a Chinese NLP classifier for legal issue triage. Given an everyday Chinese legal scenario, the system predicts one or more legal issue labels, confidence scores, and supporting law-article clues.

The project is intentionally framed as legal issue triage, not legal advice. It does not generate legal opinions, recommend litigation strategy, or replace a lawyer. Its practical goal is narrower: help a user, intake worker, or student identify likely legal topics and the statutes worth checking first.

## 2. Research Question

Can a supervised NLP pipeline map everyday Chinese legal narratives to useful multi-label legal issue categories, while remaining interpretable enough to show why a label was suggested?

The core hypothesis is that weak labels extracted from legal citations, combined with a small legal-issue ontology, are sufficient to train a baseline classifier that is more scalable than hand-written rules and more transparent than a chatbot-style answer generator.

## 3. Dataset Sources

The main weak-label source is `lianghsun/tw-legal-synthetic-qa`, a Taiwanese legal synthetic QA dataset with 9,631 examples. Each example contains a user scenario and an assistant answer. The assistant answers often include law citations, so the project uses citation normalization to derive distant-supervision labels.

The project also uses `lianghsun/tw-processed-law-article` as a statute lookup and ontology-support source. This dataset contains processed Taiwanese law articles, including law names, article text, hierarchy metadata, and update fields.

An optional out-of-domain evaluation source is `aigrant/taiwan-ly-law-research`, which contains longer legislative/legal research texts. It is not the main training distribution, but it can test whether the label space transfers beyond synthetic QA examples.

## 4. Dataset Preprocessing

The preprocessing pipeline converts raw assistant answers into structured classification examples:

1. Normalize legal citations from answer text into `(law name, article number)` pairs.
2. Map normalized citations to issue IDs through `final_project/data/issues.yaml`.
3. Keep examples with at least one mapped issue label.
4. Split usable examples into train, validation, and test sets.
5. Preserve raw citations as evidence fields for later error analysis.

The dataset usability review found that only part of the synthetic QA dataset can be safely weak-labeled by citation extraction. This is a limitation, but it is also useful: it keeps the first version grounded in observable citation evidence instead of manually inventing labels.

## 5. NLP Methods

The project compares two levels of NLP methods.

The first method is a rule-based baseline. It uses issue aliases, keyword lists, and law-article mappings from the ontology. This baseline is interpretable and easy to inspect, but it is brittle when users describe the same issue with different wording.

The second method is a TF-IDF plus one-vs-rest linear classifier. Chinese legal scenarios are represented with character n-gram features. A separate binary classifier is trained for each issue label, which fits the multi-label task because one scenario can involve several legal topics.

The planned extension is a Chinese BERT-style classifier. This may improve semantic matching, but it also increases training cost and lowers interpretability. For the course project, the baseline and TF-IDF model are the minimum meaningful comparison.

## 6. Experiment and Test Process

The experiment process is:

1. Build or update the legal issue ontology.
2. Run citation normalization and auto-labeling.
3. Split the labeled examples into train, validation, and test sets.
4. Train the rule-based baseline.
5. Train the TF-IDF linear model.
6. Evaluate all methods with the same held-out test split.
7. Review false positives and false negatives by issue type.

The main metrics are micro-F1, macro-F1, hamming loss, and per-class F1. Micro-F1 shows overall label prediction quality, while macro-F1 is important because rare legal issues should not disappear behind frequent categories.

Unit tests cover the citation normalizer, ontology YAML structure, baseline behavior, API helpers, metrics, and pipeline helper functions. CI is configured to run these Python tests and build the frontend when the web package is present.

## 7. Results to Report

| Method | Micro-F1 | Macro-F1 | Hamming Loss | Notes |
|---|---:|---:|---:|---|
| Rule-based baseline | 0.589 | 0.419 | 0.064 | High recall and interpretability, weak paraphrase handling |
| TF-IDF + one-vs-rest linear classifier | 0.760 | 0.324 | 0.025 | Better overall precision and micro-F1, weaker rare-label coverage |

Observed qualitative patterns:

- Rule-based predictions are useful when keywords or cited articles are explicit, but they over-predict because keyword matches are broad.
- TF-IDF improves overall micro-F1 and hamming loss, but macro-F1 shows that rare issue labels still need more data or better balancing.
- Rare issue labels need per-class analysis because aggregate metrics can hide weak coverage.
- In the per-label table, support `0` means the label was predicted by a model but did not appear in the held-out test labels. This is useful for spotting over-prediction.
- Out-of-domain examples are expected to perform worse than synthetic QA examples because the text style and length are different.

## 8. Discussion

The most important design decision is treating the project as classification, not legal question answering. Classification keeps the output bounded and measurable. It also avoids the higher-risk behavior of generating legal advice.

The weakest part is label quality. Citation-derived labels are practical for a student project, but they are not equivalent to expert annotation. Some answers may mention laws as background rather than as the central issue, and some valid legal issues may appear without explicit citations.

The class imbalance is another challenge. Common statutes and common issue types dominate the weak-labeled examples, so macro-F1 and per-class F1 matter more than accuracy.

## 9. Practical Application

A practical application is a legal intake assistant for educational or public-service settings. A user writes a short Chinese scenario, and the system suggests likely issue labels such as defamation, contract dispute, privacy violation, or labor dispute. The output can include confidence scores and relevant statute references for human review.

This is useful because many people do not know the legal vocabulary for their problem. The classifier can act as a first routing layer before a human expert reviews the case.

## 10. Limitations

The system does not provide legal advice and should not be used as a final decision maker.

The main training labels are weak labels extracted from synthetic QA answers, so they may contain noise. The dataset also reflects the style and assumptions of the source data rather than real-world legal intake forms.

The system may miss issues that require detailed legal interpretation, fact verification, or professional judgment. It may also perform worse on long documents, highly formal legal writing, or scenarios involving multiple jurisdictions.

## 11. Conclusion

This project shows how course NLP methods can be combined into a practical Chinese legal issue triage pipeline. Citation normalization provides weak supervision, an ontology makes labels inspectable, and multi-label classification turns everyday narratives into measurable predictions.

The final model should be judged by both quantitative metrics and error analysis. A useful system is not only the one with the highest F1 score, but the one whose mistakes are visible enough for a human to review.
