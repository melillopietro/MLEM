# Estrazione metriche — dataset originale vs v3 vs v3 (23.807 vittime)

**Data estrazione:** 2026-11-18  
**Fonte:** `experiments/comparison/summary.json` + ablation per scenario  
**CSV pronti per Excel/LaTeX:** `experiments/comparison/exports/`

| Codice export | File sorgente | Significato |
|---------------|---------------|-------------|
| `dataset_originale` | `Dataset Ransomware.xlsx`, as-of `2025-04-01` | Dataset legacy del paper |
| `v3_finestra_paper` | `DATASETv3.xlsx`, as-of `2025-04-01` (~**18.596** disclosure) | v3 allineato alla replica JISAS |
| `v3_23807_vittime` | `DATASETv3.xlsx`, **nessun** as-of (**23.807** righe Registry) | Corpus completo fino a 2025-12-31 |

---

## A. Corpus e profili

| Metrica | Dataset originale | v3 (finestra paper) | v3 (23.807) |
|---------|------------------:|--------------------:|------------:|
| Righe foglio Registry/Dataset (Excel) | 18.602 | 23.807* | 23.807 |
| Righe usate (filtro temporale) | 18.589 | 18.596 | 23.807 |
| Filtro as-of | 2025-04-01 | 2025-04-01 | — |
| Gang distinte (corpus usato) | 231 | 200 | 250 |
| Profili gang | 174 | 281 | 281 |
| Profili con tecniche ATT&CK | 129 | 129 | 129 |
| Profili con CVE documentate | 0 | 62 | 62 |
| Data massima nel corpus usato | 2025-03-31 | 2025-03-31 | 2025-12-31 |
| Disclosure anno 2025 (nel file) | 2.103 | 7.284* | 7.284 |

\*Il file v3 contiene anche il 2025 post-marzo; con il cutoff paper si usano solo ~18,6k righe.

---

## B. Tab. 5 — Ablation aggregata (Random Forest, macro-F1, split random 15% test)

| Feature set | Riferimento paper | Dataset originale | v3 (finestra paper) | v3 (23.807) |
|-------------|------------------:|------------------:|--------------------:|------------:|
| TTP only | **0,925** | 0,743 | 0,932 | 0,937 |
| TTP + contesto | **0,973** | 0,802 | 0,954 | **0,972** |
| TTP + contesto + CVE | — | 0,802 | 0,966 | **0,980** |
| Solo contesto | **0,189** | 0,043 | 0,050 | 0,062 |

| Scenario ML (soglia ≥10, TTP non vuoto) | Righe | Label |
|------------------------------------------|------:|------:|
| Dataset originale | 15.844 | 109 |
| v3 finestra paper | 12.908 | 84 |
| v3 23.807 | 16.162 | 91 |

Modelli, weighted-F1 e accuracy completi: `exports/metriche_ablation_completa.csv`.

---

## C. Audit leakage firma TTP statica

| Metrica | Dataset originale | v3 (finestra paper) | v3 (23.807) |
|---------|------------------:|--------------------:|------------:|
| Righe eleggibili | 15.844 | 12.908 | 16.181 |
| Label eleggibili | 109 | 84 | 91 |
| Firme TTP uniche | 85 | 79 | 86 |
| Firme test già in train | **1,000** | **1,000** | **1,000** |
| Lookup esatto macro-F1 | 0,727 | 0,928 | 0,934 |

---

## D. RQ3 — Classificazione cronologica (settore + paese, miglior modello per periodo)

| Dataset | RQ3 righe (dedup) | Label canoniche | Label ≥10 disp. | Con profilo ATT&CK |
|---------|------------------:|----------------:|----------------:|-------------------:|
| Originale | 18.572 | 205 | 135 | 91 |
| v3 paper | 18.561 | 200 | 133 | 84 |
| v3 23.807 | **23.747** | 250 | 172 | 91 |

| Dataset | Periodo | Modello | Closed macro-F1 | Full macro-F1 | Copertura closed-set |
|---------|---------|---------|------------------:|--------------:|---------------------:|
| Originale | validation_2024 | LogisticRegression | 0,0097 | 0,0032 | 0,638 |
| Originale | test_2025 | LogisticRegression | 0,0074 | 0,0024 | 0,510 |
| v3 paper | validation_2024 | LogisticRegression | 0,0184 | 0,0075 | 0,656 |
| v3 paper | test_2025 | LogisticRegression | 0,0082 | 0,0028 | 0,542 |
| v3 23.807 | validation_2024 | LogisticRegression | 0,0184 | 0,0075 | 0,656 |
| v3 23.807 | test_2025 | LogisticRegression | 0,0055 | 0,0010 | 0,488 |

---

## E. File export

| File | Contenuto |
|------|-----------|
| `metriche_corpus.csv` | Righe, gang, profili, date |
| `metriche_tab5_rf.csv` | Tab. 5 wide (un rigo per dataset) |
| `metriche_ablation_completa.csv` | RF + ExtraTrees, tutti i feature set |
| `metriche_signature_audit.csv` | Leakage / lookup |
| `metriche_rq3_cronologico.csv` | Metriche RQ3 per periodo |

**Rigenerazione:**

```bash
python experiments/compare_three_datasets.py   # se serve ricalcolare
python experiments/export_three_way_metrics.py # solo estrazione CSV
```

---

*Valori arrotondati a 3–4 decimali nel testo; CSV con precisione piena.*