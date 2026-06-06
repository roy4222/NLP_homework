# Acceptance Test Results

This acceptance suite is a deterministic, template-based demo regression set.
It contains 50 synthetic Chinese legal scenarios for each of the 20 labels.
It should be treated as course-demo validation, not as an independent human-labeled benchmark.

## Summary

- Total cases: 1000
- Labels tested: 20
- Labels passing threshold: 1 / 20
- Overall Top-1 accuracy: 0.794
- Overall Top-3 hit rate: 0.962
- Overall recall: 0.962

Threshold: per-label Top-1 accuracy >= 0.95, recall >= 0.95, precision >= 0.90.

## By Scenario Type

| Split | Cases | Top-1 Accuracy | Top-3 Hit Rate | Recall |
|---|---:|---:|---:|---:|
| typical | 600 | 0.857 | 1.000 | 1.000 |
| colloquial | 200 | 0.735 | 0.810 | 0.810 |
| ambiguous | 100 | 0.730 | 1.000 | 1.000 |
| near_miss | 100 | 0.600 | 1.000 | 1.000 |

## Per-label Results

| Label | Cases | Top-1 | Top-3 | Recall | Precision | Pass |
|---|---:|---:|---:|---:|---:|---:|
| `dui_public_danger` | 50 | 1.000 | 1.000 | 1.000 | 0.588 | no |
| `theft` | 50 | 0.860 | 1.000 | 1.000 | 0.417 | no |
| `aggravated_theft` | 50 | 0.220 | 0.900 | 0.900 | 1.000 | no |
| `fraud` | 50 | 0.900 | 1.000 | 1.000 | 0.588 | no |
| `injury` | 50 | 1.000 | 1.000 | 1.000 | 0.556 | no |
| `negligent_injury` | 50 | 0.460 | 1.000 | 1.000 | 0.400 | no |
| `negligent_death` | 50 | 0.800 | 0.900 | 0.900 | 0.529 | no |
| `hit_and_run` | 50 | 0.800 | 0.900 | 0.900 | 0.900 | no |
| `property_damage` | 50 | 0.700 | 0.900 | 0.900 | 0.900 | no |
| `document_forgery` | 50 | 0.800 | 1.000 | 1.000 | 1.000 | no |
| `defamation` | 50 | 1.000 | 1.000 | 1.000 | 0.769 | no |
| `intimidation` | 50 | 0.900 | 1.000 | 1.000 | 0.909 | no |
| `tort_damages` | 50 | 0.940 | 0.940 | 0.940 | 0.783 | no |
| `divorce` | 50 | 1.000 | 1.000 | 1.000 | 0.588 | no |
| `unjust_enrichment` | 50 | 0.900 | 1.000 | 1.000 | 0.588 | no |
| `contract_breach` | 50 | 0.900 | 1.000 | 1.000 | 0.625 | no |
| `sales_defect` | 50 | 0.800 | 0.900 | 0.900 | 0.818 | no |
| `inheritance` | 50 | 1.000 | 1.000 | 1.000 | 0.909 | yes |
| `marital_property` | 50 | 0.500 | 0.900 | 0.900 | 0.750 | no |
| `property_return` | 50 | 0.400 | 0.900 | 0.900 | 0.692 | no |
