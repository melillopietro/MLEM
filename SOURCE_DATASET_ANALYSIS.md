# Source workbook analysis

File: `Dataset Ransomware.xlsx`

## Sheets

| Sheet | Rows | Columns |
|---|---:|---:|
| Dataset | 18602 | 10 |
| Ransomware Gang Profile | 178 | 118 |
| MITRE ATT&CK GROUPS  | 175 | 118 |
| ID TTP | 423 | 5 |
| Dashboard | 208 | 8 |
| Note | 91 | 1 |
| CountryGang-VictimCountry | 122 | 12 |

## Dataset sheet (incidents)

- Rows: **18602**
- Distinct gangs: **231**
- Date range: **2020-01-12** – **2025-04-01**
- Unparseable dates: **0**
- Exact duplicate rows: **2**

### Disclosures by year

```json
{
  "2020": 436,
  "2021": 2616,
  "2022": 2827,
  "2023": 5287,
  "2024": 5333,
  "2025": 2103
}
```

### Top gangs

```json
{
  "lockbit3": 1974,
  "clop": 982,
  "alphv": 799,
  "play": 770,
  "ransomhub": 761,
  "akira": 693,
  "blackbasta": 564,
  "lockbit2": 533,
  "bianlian": 477,
  "8base": 458,
  "LockBit": 456,
  "Conti": 384,
  "qilin": 356,
  "Pysa": 285,
  "hunters": 276
}
```

### Top victim sectors (raw labels)

```json
{
  "Industrial": 3158,
  "Services": 2369,
  "Technology": 1687,
  "Construction": 1176,
  "Healthcare": 1116,
  "Retail": 958,
  "Public": 955,
  "Financial": 940,
  "Education": 859,
  "Hospitality": 839
}
```

### Top victim countries

```json
{
  "USA": 8920,
  "UK": 1023,
  "Canada": 910,
  "Germany": 749,
  "Italy": 632,
  "France": 590,
  "Spain": 386,
  "Brazil": 365,
  "Australia": 361,
  "India": 327
}
```

### Columns with missing values

```json
{
  "ID": 3051,
  "gang": 2,
  "Victim Country": 186,
  "Victim sectors": 307,
  "NumberN/AofN/Aemployees": 11970,
  "Sales": 12299,
  "Ransom Currency": 18601,
  "Ransom Paid": 18602
}
```

## Ransomware Gang Profile sheet

- Profile rows: **178**
- Documented gang names: **174**
- TTP column groups: **111**
- Profiles with at least one TTP token: **129**

## Pipeline role

The **Dataset** sheet feeds victim-level disclosures (gang, date, sector, country). The **Ransomware Gang Profile** sheet supplies static MITRE ATT&CK technique lists merged onto every incident with a matching gang name during `dataset_ML_Formatter.py`.
