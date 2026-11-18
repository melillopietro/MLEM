# Report comparativo: dataset legacy vs DATASETv3

**Contesti:** replicazione finestra paper (`as-of 2025-04-01`) e confronto con estensione completa v3.  
**Obiettivo:** risultati per il resubmit JISAS e mapping sulle critiche dei revisori.

Per il confronto **a tre vie** (legacy vs v3 finestra paper vs v3 completo 23.807 righe) con metriche Tab. 5 e RQ3 cronologico, vedere [`THREE_WAY_COMPARISON_REPORT.md`](THREE_WAY_COMPARISON_REPORT.md).

---

## 1. Executive summary

| Conclusione | Dettaglio |
|-------------|-----------|
| Corpus disclosure 2020–Q1 2025 | **Quasi identico** (~18.6k righe); differenze minori su duplicati e normalizzazione. |
| Valore aggiunto v3 | **Profili estesi**, **CVE documentate**, **metadati CTI**, **merge actor_key**, **tooling replication** — non un rivolgimento dei numeri cronologici. |
| Revisioni metodologiche | v3 **non elimina** il leakage TTP; lo **rende auditabile** (`signature_overlap_audit`, ablation con CVE). |
| Estensione oltre aprile 2025 | **+5.181 disclosure** nel solo 2025 (v3 vs legacy). |

---

## 2. Confronto corpus (finestra paper `2025-04-01`)

| Metrica | Legacy (`Dataset Ransomware.xlsx`) | DATASETv3 | Δ |
|---------|-------------------------------------|-----------|---|
| Righe registry | 18.589 | 18.596 | +7 |
| Gang distinte (grezze) | 231 | 200 | −31 (canonicalizzazione/spelling) |
| Range date max | 2025-03-31 | 2025-03-31 | = |
| Disclosure 2025 | 2.090 | 2.097 | +7 |
| Missing paese | 186 | 0 | Pulizia v3 |
| Missing settore | 306 | 0 | Pulizia v3 |
| Duplicati esatti | 2 | 24 | Più righe duplicate segnalate |
| Profili gang | 174 | **281** | +107 |
| Profili con tecniche ATT&CK | 128 | 129 | ≈ |
| Profili con CVE documentate | **0** | **62** | Nuovo in v3 |
| Firme TTP uniche (profili) | 119 | 121 | ≈ |

### Estensione temporale (solo v3)

| Anno | Legacy | v3 full | Δ |
|------|--------|---------|---|
| 2025 completo | 2.103 | 7.284 | **+5.181** |
| 2020–2024 | invariato | invariato | 0 |

---

## 3. Pipeline ML aggregata (split random, ≥10 campioni, con TTP)

| Metrica | Legacy | DATASETv3 | Interpretazione |
|---------|--------|-----------|-----------------|
| Righe ML encoded | 15.983 | 13.323* | v3 più selettivo (TTP non vuoti + date) |
| Feature | 448 | **526** | +metadati profilo, CVE, settore/paese |
| Label eleggibili (≥10) | **109** | **84** | Meno overlap varianti nome in v3 |
| Firme TTP uniche (label) | 85 | 79 | Meno collisioni da varianti case |
| **RF macro-F1 TTP only** | 0,743 | **0,932** | Più vicino al paper (0,925); audit leakage necessario |
| **RF macro-F1 TTP+contesto** | 0,802 | **0,954** | Vicino Tab. 5 paper (0,973) |
| **RF macro-F1 + CVE** | 0,802 (no CVE) | **0,966** | **Risponde Rev.2**: CVE negli esperimenti |
| Lookup firma esatta (test) | 0,727 | **0,928** | Entrambi con 100% firme test già in train |

\*Dalla replica `run_replication.py paper_jisas2026`.

---

## 4. RQ3 / cronologico (settore + paese)

| Metrica | Legacy (paper originale) | v3 (replica paper) |
|---------|--------------------------|---------------------|
| Righe deduplicate | 18.585 → **18.572** | **18.561** |
| Label canoniche | 205 | 200 |
| Label ≥10 disclosure | **135** | 133 |
| Con profilo ATT&CK | 91 | 84 |
| Val. 2024 closed macro-F1 (LR) | 0,0097 | **0,018** |
| Test 2025 frozen closed F1 | 0,0074 | **0,008** |
| Test 2025 copertura frozen | 50,6% | **54,2%** |

**Messaggio:** la conclusione del paper (cronologico **non** attribuisce) **resta valida** con v3; leggero miglioramento numerico sulla validazione 2024 non cambia l’ordine di grandezza.

---

## 5. Paper JISAS — Tab. 5 (riferimento vs v3 replica)

| Configurazione | Paper (RF) | v3 replica |
|----------------|------------|------------|
| TTP only | 0,9247 | **0,932** |
| TTP + contesto | 0,9726 | **0,954** |
| Context only | 0,1892 | **0,050** |

Contesto-only diverge per etichette/spelling; TTP e combinato restano nello stesso ordine di grandezza.

---

## 6. Cosa otteniamo in più con DATASETv3

### Dati e profili
- **Registry** più pulito (paese/settore completi nel foglio v3).
- **281 profili** con origine, extortion type, lineage, status, fase ecosistema, impatto LE.
- **TTP_Techniques / TTP_Tactics** espliciti (non 111 colonne sparse).
- **62 gang** con CVE documentate → ablation e discussione CTI.

### Metodo e replica (risposta revisori)
| Critica revisore | Artefatto MLEM con v3 |
|------------------|------------------------|
| Statistiche dataset assenti | `reports/DATASET_DESCRIPTOR.md` |
| Pipeline TTP/CVE non descritta | `mlem_workbook.py`, `ACTOR_LABEL_POLICY.md` |
| Leakage / Tab. 5 | `signature_overlap_audit.json` |
| CVE non in esperimenti | `aggregate_ablation_results.csv` (+CVE row) |
| Baselines cronologiche | `MajorityBaseline` in RQ3 outputs |
| Naming / rebranding | `actor_key`, mapping RQ3, policy LockBit2/3 |
| Replica paper | `run_replication.py paper_jisas2026` |

### Analisi oltre il paper
- Timeline fino a **31/12/2025** (+5.2k eventi nel 2025).
- **Qilin**, aggiornamenti conteggio gang, `Medusalocker`, spelling normalizzato.
- Feature space **548** (extended) vs **503** legacy per studi profilo/CVE aggiuntivi.

---

## 7. Cosa v3 **non** risolve da solo

1. **Leakage firma TTP statica** — resta; serve linguaggio onesto + audit (implementato).
2. **Attribuzione cronologica** — macro-F1 ~0,01 invariato.
3. **Coerenza narrativa RQ1** (10 vs 6 gang, LockBit versioni) — serve testo, non solo dati.
4. **TTP/CVE time-stamped** — non disponibili per OOT tecnico.

---

## 8. Raccomandazione per il resubmit

1. **Usare v3** come source of truth del replication package.
2. Nel manoscritto mantenere **due finestre**: (A) paper fino 2025-04-01, (B) supplemento 2025 pieno.
3. Citare **tre numeri-chiave aggiornati** con v3: descriptor, ablation +CVE, signature audit.
4. **Non** rivendicare miglioramento attribution da v3; rivendicare **copertura CTI, trasparenza, CVE in ablation**.

---

*Generato da confronti eseguiti su MLEM branch `cursor/fix-pipeline-and-analyze-9d8e`.*