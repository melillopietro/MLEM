# Confronto a tre vie: legacy vs v3 (paper) vs v3 (completo)

**Generato:** 2026-11-18  
**Scenari:** (1) `Dataset Ransomware.xlsx` con cutoff paper; (2) `DATASETv3.xlsx` stesso cutoff;
(3) `DATASETv3.xlsx` senza cutoff (tutte le righe Registry, fino a 2025-12-31).

Riferimento Tab. 5 paper (Random Forest, split aggregato random): TTP **0,925**, TTP+contesto **0,973**, solo contesto **0,189**.

---

## 1. Corpus

| Scenario | Righe Excel | Dopo as-of | Gang (as-of) | Profili | Data max (file) | Data max (as-of) |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy (finestra paper) | 18602 | 18589 | 231 | 174 | 2025-04-01 | 2025-03-31 |
| DATASETv3 finestra paper | 23807 | 18596 | 200 | 281 | 2025-12-31 | 2025-03-31 |
| DATASETv3 completo (23.807 righe) | 23807 | 23807 | 250 | 281 | 2025-12-31 | 2025-12-31 |

**Lettura:** ~**18,6k** disclosure compaiono solo con cutoff **2025-04-01** (legacy ≈ v3 paper).
Il terzo scenario aggiunge **~5,2k** eventi (2025 apr–dic); le righe Excel passano a **23.807**.

---

## 2. Ablation aggregata (Random Forest macro-F1, come Tab. 5)

| Feature set | Paper RF | Legacy paper | v3 paper | v3 full |
| --- | --- | --- | --- | --- |
| ttp_only | 0.925 | 0.743 | 0.932 | 0.937 |
| ttp_plus_context | 0.973 | 0.802 | 0.954 | 0.972 |
| ttp_plus_context_plus_cve | — | 0.802 | 0.966 | 0.980 |
| context_only | 0.189 | 0.043 | 0.050 | 0.062 |

Metriche ausiliarie (righe/label ML con TTP non vuoto, soglia ≥10):

| Scenario | Righe ML | Label (≥10) |
|----------|---------:|------------:|
| Legacy paper | 15844 | 109 |
| v3 paper | 12908 | 84 |
| v3 full | 16162 | 91 |

---

## 3. Audit leakage firma TTP (split random 15% test)

| Scenario | Righe eleggibili | Label | Firme test già in train | Lookup macro-F1 |
| --- | --- | --- | --- | --- |
| Legacy (finestra paper) | 15844 | 109 | 1.000 | 0.727 |
| DATASETv3 finestra paper | 12908 | 84 | 1.000 | 0.928 |
| DATASETv3 completo (23.807 righe) | 16181 | 91 | 1.000 | 0.934 |

---

## 4. RQ3 cronologico (miglior modello per periodo, feature sector+country)

| Scenario | Periodo | Modello | Closed macro-F1 | Copertura closed-set |
| --- | --- | --- | --- | --- |
| Legacy (finestra paper) | validation_2024 | LogisticRegression | 0.0097 | 0.638 |
| Legacy (finestra paper) | test_2025 | LogisticRegression | 0.0074 | 0.510 |
| DATASETv3 finestra paper | validation_2024 | LogisticRegression | 0.0184 | 0.656 |
| DATASETv3 finestra paper | test_2025 | LogisticRegression | 0.0082 | 0.542 |
| DATASETv3 completo (23.807 righe) | validation_2024 | LogisticRegression | 0.0184 | 0.656 |
| DATASETv3 completo (23.807 righe) | test_2025 | LogisticRegression | 0.0055 | 0.488 |

**Messaggio per il paper:** estendere il corpus a **23,8k** **non** riporta l’attribuzione cronologica
verso i valori Tab. 5; i numeri restano nell’ordine di **10⁻²–10⁻³**. Il salto verso ~0,93–0,97
resta confinato al benchmark **aggregato** con profili TTP statici.

---

## 5. Implicazioni resubmit

| Domanda revisore | Cosa mostrano i 3 scenari |
|------------------|---------------------------|
| Dataset stats | Riga **v3 full** documenta l’estensione 2025 senza confonderla con la finestra paper |
| Tab. 5 / leakage | Legacy vs **v3 paper**: profili/CVE migliorano RF aggregato; lookup ~100% firme in train in tutti |
| Attribution OOT | **v3 full** aumenta il rumore temporale ma **non** il macro-F1 cronologico |
| Quale corpus citare | Manoscritto principale: **v3 paper** (parità legacy); supplemento: **v3 full** |

Artefatti per scenario: `experiments/comparison/<scenario_id>/`.

---

*Pipeline: `python3 experiments/compare_three_datasets.py`*
