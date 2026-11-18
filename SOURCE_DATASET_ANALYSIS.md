# Source workbook analysis

File: `DATASETv3.xlsx` (schema **v3**)

## Sheets

| Sheet | Rows | Columns |
|---|---:|---:|
| Registry | 23807 | 5 |
| Gang Profile | 281 | 21 |

## Registry / Dataset sheet

- Rows: **18596**
- Distinct gangs: **200**
- Date range: **2020-01-12** – **2025-03-31**
- Unparseable dates: **0**
- Exact duplicate rows: **24**

### Disclosures by year

```json
{
  "2020": 436,
  "2021": 2616,
  "2022": 2827,
  "2023": 5287,
  "2024": 5333,
  "2025": 2097
}
```

### Top gangs

```json
{
  "Lockbit3": 1974,
  "Clop": 1088,
  "Alphv": 807,
  "Play": 770,
  "Ransomhub": 759,
  "Akira": 693,
  "Blackbasta": 564,
  "Conti": 537,
  "Lockbit2": 533,
  "Bianlian": 477,
  "Medusalocker": 467,
  "8Base": 458,
  "Lockbit": 456,
  "Qilin": 356,
  "Pysa": 285
}
```

### Columns with missing values

```json
{}
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
