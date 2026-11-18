# Source workbook analysis

File: `DATASETv3.xlsx` (schema **v3**)

## Sheets

| Sheet | Rows | Columns |
|---|---:|---:|
| Registry | 23807 | 5 |
| Gang Profile | 281 | 21 |

## Registry / Dataset sheet

- Rows: **23807**
- Distinct gangs: **250**
- Date range: **2020-01-12** – **2025-12-31**
- Unparseable dates: **24**
- Exact duplicate rows: **24**

### Disclosures by year

```json
{
  "2020.0": 436,
  "2021.0": 2616,
  "2022.0": 2827,
  "2023.0": 5287,
  "2024.0": 5333,
  "2025.0": 7284
}
```

### Top gangs

```json
{
  "Lockbit3": 1995,
  "Qilin": 1264,
  "Clop": 1210,
  "Akira": 1199,
  "Play": 1070,
  "Alphv": 807,
  "Ransomhub": 761,
  "Blackbasta": 564,
  "Medusalocker": 549,
  "Conti": 537,
  "Lockbit2": 533,
  "Incransom": 488,
  "Bianlian": 477,
  "8Base": 458,
  "Lockbit": 456
}
```

### Columns with missing values

```json
{
  "date": 24
}
```

## Gang profile sheet

- Profile rows: **281**
- Documented gang names: **281**
- Profiles with ATT&CK techniques: **129**
- Profiles with ATT&CK tactics: **14**

### Profile column groups

```json
[
  "gang",
  "profile_registry_flag",
  "ttp_tactics",
  "ttp_techniques",
  "profile_primary_motivation",
  "profile_origin",
  "profile_extortion_type",
  "cves_exploited",
  "profile_programming_language",
  "profile_currency",
  "First_Appearance",
  "First_Appearance_in_Dataset",
  "Last_Appearance",
  "profile_current_status",
  "profile_lineage_parent",
  "profile_lineage_type",
  "profile_confidence_level",
  "profile_law_enforcement_impact",
  "profile_ecosystem_phase",
  "profile_dataset_event_correlation",
  "profile_aliases",
  "actor_key"
]
```

## Pipeline role

Disclosure rows feed `Dataset Normalized.csv`. Gang profiles supply `TTP_Techniques` / `TTP_Tactics` (v3) or legacy wide TTPS columns, plus metadata (origin, extortion type, CVE list, lineage, status) merged in `dataset_ML_Formatter.py` using a case-insensitive `actor_key`.
