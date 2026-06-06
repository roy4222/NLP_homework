# Experimental Results

| Method | Micro-F1 | Macro-F1 | Precision | Recall | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Rule-based | 0.589 | 0.419 | 0.457 | 0.831 | 0.064 |
| TF-IDF + SVM | 0.760 | 0.324 | 0.851 | 0.686 | 0.025 |

## Per-label F1

### Rule-based

| Label | Support | F1 |
|---|---:|---:|
| `inheritance` | 0 | 0.000 |
| `marital_property` | 1 | 0.000 |
| `negligent_death` | 5 | 0.148 |
| `contract_breach` | 2 | 0.182 |
| `hit_and_run` | 2 | 0.190 |
| `unjust_enrichment` | 6 | 0.250 |
| `injury` | 14 | 0.286 |
| `document_forgery` | 1 | 0.333 |
| `aggravated_theft` | 3 | 0.333 |
| `property_damage` | 6 | 0.357 |
| `negligent_injury` | 20 | 0.424 |
| `tort_damages` | 30 | 0.481 |
| `intimidation` | 5 | 0.556 |
| `sales_defect` | 5 | 0.600 |
| `defamation` | 6 | 0.625 |
| `fraud` | 12 | 0.690 |
| `divorce` | 7 | 0.778 |
| `theft` | 15 | 0.789 |
| `dui_public_danger` | 102 | 0.936 |

### TF-IDF + SVM

| Label | Support | F1 |
|---|---:|---:|
| `document_forgery` | 1 | 0.000 |
| `marital_property` | 1 | 0.000 |
| `contract_breach` | 2 | 0.000 |
| `hit_and_run` | 2 | 0.000 |
| `aggravated_theft` | 3 | 0.000 |
| `intimidation` | 5 | 0.000 |
| `negligent_death` | 5 | 0.000 |
| `property_damage` | 6 | 0.000 |
| `unjust_enrichment` | 6 | 0.000 |
| `defamation` | 6 | 0.286 |
| `sales_defect` | 5 | 0.500 |
| `fraud` | 12 | 0.522 |
| `injury` | 14 | 0.526 |
| `tort_damages` | 30 | 0.536 |
| `theft` | 15 | 0.788 |
| `divorce` | 7 | 0.824 |
| `negligent_injury` | 20 | 0.865 |
| `dui_public_danger` | 102 | 0.985 |
