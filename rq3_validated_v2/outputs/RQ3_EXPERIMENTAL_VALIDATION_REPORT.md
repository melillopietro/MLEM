# RQ3 experimental validation report

## Analytical boundary

All-time ATT&CK profiles are restricted to the aggregated actor-profile baseline. They are excluded from out-of-time classification because they are not versioned by year and may contain knowledge observed after a training cutoff. Temporal classification uses only disclosure-level sector and country as its primary feature set.

## Dataset audit

- **raw_rows**: 23807
- **rows_with_valid_label_and_date**: 23783
- **deduplicated_rows**: 23747
- **exact_duplicate_rows_removed_before_label_normalisation**: 36
- **additional_duplicate_rows_removed_after_label_normalisation**: 0
- **raw_actor_labels**: 250
- **canonical_actor_labels**: 250
- **lexical_label_variants_remapped**: 21
- **eligible_labels_min_10_disclosures**: 172
- **eligible_labels_with_valid_attack_profile**: 91
- **eligible_labels_without_valid_attack_profile**: ['8base', 'Akira', 'Alphv', 'Apt73 Bashe', 'Argonauts', 'Astroteam', 'Babuk2', 'Beast', 'Blacklock', 'Blacknevas', 'Blackshrantac', 'Brotherhood', 'Cephalus', 'Chaos', 'Coomingproject', 'Crypto24', 'D4RK4RMY', 'Darkside', 'Datacarry', 'Direwolf', 'Doppelpaymer', 'Egregor', 'Flocker', 'Frag', 'Freecivilian', 'Genesis', 'Global', 'Grief', 'Handala', 'Haron', 'Hellcat', 'Hiveleak', 'Icefire', 'Imncrew', 'Interlock', 'J', 'Kairos', 'Knight', 'Kraken', 'Lockbit5', 'Lvblog', 'Mad Liberator', 'Marketo', 'Meow', 'Mindware', 'Moneymessage', 'Morpheus', 'Mydata', 'Netwalker', 'Nightspire', 'Nitrogen', 'Nova', 'Obscura', 'Payoutsking', 'Pear', 'Prometheus', 'Pysa', 'Radar', 'Ragnarock', 'Ralord', 'Ransom', 'Ransomedvc', 'Ransomexx', 'Ransomhub', 'Redransomware', 'Royal', 'Silentransomgroup', 'Snatch', 'Spacebears', 'Sparta', 'Teamxxx', 'Termite', 'Thegentlemen', 'Threeam', 'Tridentlocker', 'Unknown', 'Vicesociety', 'Warlock', 'Werewolves', 'Weyhro', 'Worldleaks']
- **invalid_technical_token_rows**: 0
- **excluded_tactic_token_rows**: 71
- **date_min**: 2020-01-12
- **date_max**: 2025-12-31

## Main temporal split

- Training: **2020–2023** (10982 eligible records)
- Validation: **2024** (3496 known; 1837 unseen-label records)
- Test: **2025 through 2025-12-31** (3551 known; 3732 unseen-label records)
- The 2025 test interval is partial.

| Period | Model | Coverage | Closed macro-F1 | Full macro-F1 | Closed weighted-F1 | Full accuracy |
|---|---|---:|---:|---:|---:|---:|
| validation_2024 | MajorityBaseline | 0.6555 | 0.0067 | 0.0019 | 0.0394 | 0.0986 |
| validation_2024 | LogisticRegression | 0.6555 | 0.0184 | 0.0075 | 0.0451 | 0.0874 |
| validation_2024 | LogisticRegressionBalanced | 0.6555 | 0.0066 | 0.0030 | 0.0093 | 0.0069 |
| validation_2024 | RandomForestUnweighted | 0.6555 | 0.0174 | 0.0071 | 0.0438 | 0.0778 |
| validation_2024 | RandomForest | 0.6555 | 0.0067 | 0.0031 | 0.0126 | 0.0084 |
| test_2025 | MajorityBaseline | 0.4876 | 0.0008 | 0.0001 | 0.0003 | 0.0058 |
| test_2025 | LogisticRegression | 0.4876 | 0.0055 | 0.0010 | 0.0284 | 0.0181 |

## 2025 contextual ablation

| Feature set | Coverage | Closed macro-F1 | Full macro-F1 |
|---|---:|---:|---:|
| sector_only | 0.4876 | 0.0027 | 0.0004 |
| country_only | 0.4876 | 0.0013 | 0.0003 |
| sector_country | 0.4876 | 0.0055 | 0.0010 |
| sector_country_calendar_sensitivity | 0.4876 | 0.0007 | 0.0002 |

## Rolling-origin evaluation

| Test year | Train through | Coverage | Closed macro-F1 | Full macro-F1 | Unseen records |
|---:|---:|---:|---:|---:|---:|
| 2021 | 2020 | 0.4340 | 0.0647 | 0.0083 | 1479 |
| 2022 | 2021 | 0.3048 | 0.0158 | 0.0032 | 1964 |
| 2023 | 2022 | 0.6483 | 0.0240 | 0.0091 | 1849 |
| 2024 | 2023 | 0.6555 | 0.0184 | 0.0075 | 1837 |
| 2025 | 2024 | 0.7409 | 0.0140 | 0.0061 | 1887 |

## Frozen versus updated model on partial 2025

| Policy | Coverage | Closed macro-F1 | Full macro-F1 | Δ closed macro-F1 |
|---|---:|---:|---:|---:|
| frozen_2020_2023 | 0.4876 | 0.0055 | 0.0010 | 0.0000 |
| updated_2020_2024 | 0.7409 | 0.0140 | 0.0061 | 0.0085 |

## Contextual drift outputs

- Adjacent gang-year comparisons: **328**
- Comparisons meeting the minimum chi-square rule: **266**
- Sector and country are reported separately with entropy, top-1/top-3 concentration, Jensen–Shannon divergence, chi-square and Cramér's V.

## Interpretation boundary

Closed-set metrics measure differentiation among actor labels eligible in the training interval. Full temporal metrics additionally penalise records belonging to unseen labels, which the supervised classifier cannot emit. A difference between the two therefore reflects actor turnover as well as classification error. No technical-drift claim can be derived from the all-time ATT&CK profiles.
