# RQ3 experimental validation report

## Analytical boundary

All-time ATT&CK profiles are restricted to the aggregated actor-profile baseline. They are excluded from out-of-time classification because they are not versioned by year and may contain knowledge observed after a training cutoff. Temporal classification uses only disclosure-level sector and country as its primary feature set.

## Dataset audit

- **raw_rows**: 18602
- **rows_with_valid_label_and_date**: 18600
- **deduplicated_rows**: 18585
- **exact_duplicate_rows_removed_before_label_normalisation**: 15
- **additional_duplicate_rows_removed_after_label_normalisation**: 0
- **raw_actor_labels**: 231
- **canonical_actor_labels**: 205
- **lexical_label_variants_remapped**: 44
- **eligible_labels_min_10_disclosures**: 135
- **eligible_labels_with_valid_attack_profile**: 91
- **eligible_labels_without_valid_attack_profile**: ['AstroTeam', 'BrainCipher', 'CoomingProject', 'DarkSide', 'DoppelPaymer', 'Egregor', 'Grief', 'Haron', 'Hive', 'LVBlog', 'LockBit', 'Marketo', 'NetWalker', 'Prometheus', 'Ragnarock', 'Unknown', 'apt73/bashe', 'argonauts', 'babuk2', 'flocker', 'frag', 'freecivilian', 'funksec', 'handala', 'icefire', 'interlock', 'kairos', 'knight', 'madliberator', 'medusalocker', 'medusalockerlocker', 'moneymessage', 'mydata', 'nightspire', 'nitrogen', 'ransom', 'ransomedvc', 'raworld', 'redransomware', 'sarcoma', 'spacebears', 'sparta', 'termite', 'werewolves']
- **invalid_technical_token_rows**: 54
- **excluded_tactic_token_rows**: 148
- **date_min**: 2020-01-12
- **date_max**: 2025-04-01

## Main temporal split

- Training: **2020–2023** (10991 eligible records)
- Validation: **2024** (3404 known; 1929 unseen-label records)
- Test: **2025 through 2025-04-01** (1064 known; 1037 unseen-label records)
- The 2025 test interval is partial.

| Period | Model | Coverage | Closed macro-F1 | Full macro-F1 | Closed weighted-F1 | Full accuracy |
|---|---|---:|---:|---:|---:|---:|
| validation_2024 | MajorityBaseline | 0.6383 | 0.0069 | 0.0018 | 0.0414 | 0.0986 |
| validation_2024 | LogisticRegression | 0.6383 | 0.0097 | 0.0032 | 0.0482 | 0.0896 |
| validation_2024 | LogisticRegressionBalanced | 0.6383 | 0.0055 | 0.0024 | 0.0109 | 0.0075 |
| validation_2024 | RandomForestUnweighted | 0.6383 | 0.0095 | 0.0032 | 0.0472 | 0.0797 |
| validation_2024 | RandomForest | 0.6383 | 0.0064 | 0.0028 | 0.0101 | 0.0069 |
| test_2025 | MajorityBaseline | 0.5064 | 0.0015 | 0.0003 | 0.0008 | 0.0100 |
| test_2025 | LogisticRegression | 0.5064 | 0.0074 | 0.0023 | 0.0739 | 0.0343 |

## 2025 contextual ablation

| Feature set | Coverage | Closed macro-F1 | Full macro-F1 |
|---|---:|---:|---:|
| sector_only | 0.5064 | 0.0047 | 0.0012 |
| country_only | 0.5064 | 0.0022 | 0.0006 |
| sector_country | 0.5064 | 0.0074 | 0.0023 |
| sector_country_calendar_sensitivity | 0.5064 | 0.0063 | 0.0017 |

## Rolling-origin evaluation

| Test year | Train through | Coverage | Closed macro-F1 | Full macro-F1 | Unseen records |
|---:|---:|---:|---:|---:|---:|
| 2021 | 2020 | 0.4338 | 0.0647 | 0.0083 | 1480 |
| 2022 | 2021 | 0.3047 | 0.0159 | 0.0032 | 1965 |
| 2023 | 2022 | 0.6176 | 0.0243 | 0.0088 | 2017 |
| 2024 | 2023 | 0.6383 | 0.0097 | 0.0032 | 1929 |
| 2025 | 2024 | 0.9034 | 0.0189 | 0.0126 | 203 |

## Frozen versus updated model on partial 2025

| Policy | Coverage | Closed macro-F1 | Full macro-F1 | Δ closed macro-F1 |
|---|---:|---:|---:|---:|
| frozen_2020_2023 | 0.5064 | 0.0074 | 0.0023 | 0.0000 |
| updated_2020_2024 | 0.9034 | 0.0189 | 0.0126 | 0.0115 |

## Contextual drift outputs

- Adjacent gang-year comparisons: **316**
- Comparisons meeting the minimum chi-square rule: **246**
- Sector and country are reported separately with entropy, top-1/top-3 concentration, Jensen–Shannon divergence, chi-square and Cramér's V.

## Interpretation boundary

Closed-set metrics measure differentiation among actor labels eligible in the training interval. Full temporal metrics additionally penalise records belonging to unseen labels, which the supervised classifier cannot emit. A difference between the two therefore reflects actor turnover as well as classification error. No technical-drift claim can be derived from the all-time ATT&CK profiles.
