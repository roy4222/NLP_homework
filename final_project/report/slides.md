# Chinese Legal Issue Triage

8-10 minute English oral presentation outline and speaker notes.

## Slide 1: Title

**Chinese Legal Issue Triage: Multi-label Prediction from Everyday Legal Narratives**

- NLP Final Project
- Topic: Text Classification
- Demo: Flask API + Next.js interface

Speaker notes, about 45 seconds:
Good morning everyone. My final project is called Chinese Legal Issue Triage. The goal is to classify everyday Chinese legal scenarios into one or more legal issue labels. I want to emphasize from the beginning that this is not a legal advice chatbot. It does not generate legal opinions. Instead, it performs a smaller and more measurable NLP task: given a user's Chinese description, predict likely legal issue categories such as DUI, fraud, injury, divorce, tort damages, or contract breach.

## Slide 2: Research Question

**Can we map everyday Chinese legal narratives to useful legal issue labels?**

- Input: a short Chinese legal scenario
- Output: multiple possible legal issue labels
- Use case: intake triage before search, consultation, or human review

Speaker notes, about 55 seconds:
The research question is whether a supervised NLP pipeline can map everyday Chinese legal narratives to useful multi-label legal issue categories. This matters because normal users usually do not describe their problems with legal vocabulary. For example, a user may say, "Someone used my photo on Threads and insulted people," instead of saying "defamation" or "identity misuse." A classifier can act as an early routing layer. It helps decide what kind of legal topic the case may involve before any retrieval system, chatbot, or lawyer consultation happens.

## Slide 3: Practical Motivation

- Existing legal AI products often focus on question answering, judgment search, contract review, or document generation.
- Those systems still need issue understanding at the beginning.
- A bounded classifier is safer and easier to evaluate than a legal advice generator.

Speaker notes, about 60 seconds:
When I looked at legal AI products in Taiwan, many of them focus on legal question answering, judgment retrieval, contract review, or document generation. These are useful, but they are also large product-level systems. For a course project, building a weak version of a legal chatbot would be hard to evaluate and easy to overclaim. So I narrowed the scope. This project focuses on a component that those larger systems also need: issue triage. The classifier does not answer the legal question. It only predicts likely issue labels and related statute clues for human review.

## Slide 4: Dataset Sources

- Main dataset: `lianghsun/tw-legal-synthetic-qa`
- Auxiliary statute dataset: `lianghsun/tw-processed-law-article`
- Evaluated but not used as main data: `tw-processed-law-ctx`, `tw-legal-nlp`, `tw-legal-benchmark-v1`

Speaker notes, about 65 seconds:
The main dataset is `tw-legal-synthetic-qa`. It contains 9,631 examples in a message format, with a user legal scenario and an assistant legal analysis. I use the user message as the input text. I do not directly use the assistant answer as a generated response. Instead, I extract legal citations from it and use those citations as weak supervision. The auxiliary dataset is `tw-processed-law-article`, which contains processed Taiwan statute articles. I use it to check law names, article numbers, and ontology mappings. Other datasets were reviewed, but they were either too small, too long-context oriented, or shaped like legal benchmark questions instead of scenario classification data.

## Slide 5: Label Design

- Do not classify exact articles directly.
- Do not classify only broad law names.
- Use a middle layer: legal issue labels.

Examples:

- `刑法第185條之3` → DUI / public danger
- `民法第184條` → tort damages
- `刑法第339條` → fraud

Speaker notes, about 70 seconds:
The most important design decision is label granularity. Exact article classification is too sparse. There are many possible law articles, and many of them appear only a few times. On the other hand, classifying only broad law names, such as Criminal Code or Civil Code, is too coarse and not very useful for users. So I use a middle layer: legal issue labels. For example, Criminal Code article 185-3 maps to DUI or public danger. Civil Code article 184 maps to tort damages. Criminal Code article 339 maps to fraud. This makes the task more practical and more learnable.

## Slide 6: Preprocessing Pipeline

1. Extract citations from assistant answers.
2. Normalize formats such as `刑法第185條之3`.
3. Map citations to issue IDs through `issues.yaml`.
4. Keep examples with at least one mapped label.
5. Split into train, validation, and test sets.

Speaker notes, about 60 seconds:
The preprocessing pipeline starts with citation extraction. Chinese legal citations can appear in different formats, so I normalize them into stable law and article keys. Then I map those keys to issue labels using a manually defined ontology file. If a scenario has at least one mapped issue, it becomes a weakly labeled training example. This produced 1,523 usable examples. The remaining examples were discarded because they did not map clearly to the current MVP label set. This is a limitation, but it keeps the first version grounded in observable citation evidence.

## Slide 7: NLP Methods

- Rule-based baseline:
  - aliases and keywords from ontology
  - highly interpretable
- TF-IDF + one-vs-rest SVM:
  - character n-gram features
  - multi-label linear classifiers
- BERT:
  - planned extension if time allows

Speaker notes, about 60 seconds:
I compare two main methods. The first is a rule-based baseline using aliases and keywords from the ontology. It is easy to inspect and explain, but it is brittle when the user uses unexpected wording. The second method is TF-IDF plus one-vs-rest SVM. I use Chinese character n-grams because they work reasonably well without relying completely on word segmentation. Since this is a multi-label task, each issue label is treated as a separate binary classification problem. BERT is listed as a future extension, but the core comparison already satisfies the course topic of text classification.

## Slide 8: Experiment Process

- Build weak labels from citations.
- Train rule-based and TF-IDF/SVM models.
- Evaluate on the same held-out test split.
- Report micro-F1, macro-F1, precision, recall, and hamming loss.
- Inspect errors by issue type.

Speaker notes, about 55 seconds:
The experiment process is designed to be reproducible. First, I build weak labels from citations. Then I split the data and train both methods. Both models are evaluated on the same held-out test set. I report micro-F1 because it shows overall performance, and macro-F1 because the label distribution is imbalanced. Hamming loss is also useful because this is a multi-label problem. Finally, I inspect false positives and false negatives to understand which labels are easy or difficult.

## Slide 9: Results

| Method | Micro-F1 | Macro-F1 | Hamming Loss |
|---|---:|---:|---:|
| Rule-based baseline | 0.589 | 0.419 | 0.064 |
| TF-IDF linear classifier | 0.760 | 0.324 | 0.025 |

Speaker notes, about 70 seconds:
The TF-IDF model improves micro-F1 from 0.589 to 0.760 and lowers hamming loss from 0.064 to 0.025. This means it performs better overall and makes fewer label-level mistakes. However, the macro-F1 is lower than the rule-based baseline. This tells us that the statistical model is stronger on frequent labels but still struggles with rare categories. This is an important result because it shows why aggregate performance is not enough. For a legal triage system, rare issues still matter, so the next improvement should focus on label balance and better examples for rare categories.

## Slide 10: Interactive Demo

- Flask API: `/api/predict`
- Next.js frontend with shadcn-style UI
- Shows:
  - predicted issue labels
  - confidence scores
  - supporting law articles
  - matched keywords
  - "not legal advice" disclaimer

Speaker notes, about 60 seconds:
The demo is a small web interface. The backend is a Flask API, and the frontend is built with Next.js. The user enters a Chinese legal scenario, and the interface shows predicted issue labels, confidence scores, supporting law articles, and matched keywords. The supporting articles are shown only as lookup clues, not as legal advice. During the demo, I can enter a DUI scenario and show that the system predicts DUI or public danger with Criminal Code article 185-3 as a supporting clue.

## Slide 11: Limitations

- Labels are weak labels, not expert annotations.
- The current label set is limited to 20 MVP issues.
- The dataset is biased toward frequent criminal and civil topics.
- Procedural law labels are excluded from version one.
- The tool cannot verify facts or legal applicability.

Speaker notes, about 65 seconds:
There are several limitations. First, the labels are weak labels derived from citations in synthetic answers, not expert human annotations. Second, the MVP label set has only 20 issue categories, so it does not cover all areas of law. Third, the data is biased toward frequent criminal and civil topics. I also exclude procedural law labels such as simplified judgments or dismissal procedures because this project focuses on everyday issue triage, not court procedure classification. Finally, the tool cannot verify facts or decide whether a statute truly applies to a case.

## Slide 12: Conclusion

- The project builds a focused Chinese NLP classification pipeline.
- Issue triage is more measurable than legal answer generation.
- TF-IDF/SVM improves overall performance over rules.
- Future work: expert labels, BERT, OOD testing, better rare-label coverage.

Speaker notes, about 55 seconds:
To conclude, this project demonstrates a focused Chinese NLP pipeline for legal issue triage. The main contribution is not a legal chatbot, but a measurable classification system that maps everyday legal narratives to structured issue labels. The TF-IDF model improves overall performance over the rule-based baseline, while the macro-F1 result shows that rare labels remain difficult. Future work should include expert annotation, BERT fine-tuning, out-of-domain testing, and more balanced data for rare issue categories. Thank you.
