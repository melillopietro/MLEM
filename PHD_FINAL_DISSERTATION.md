# Strategie Avanzate di Machine Learning per l'Attribuzione Ransomware
**Autore:** Pietro Melillo | **Data:** 08/12/2025
**Corso:** Machine Learning and Empirical Methods (MLEM)

---

## 1. Abstract
Questo studio affronta il problema dell'attribuzione degli attacchi ransomware, proponendo un framework di classificazione multiclasse basato sull'analisi delle Tattiche, Tecniche e Procedure (TTP) del framework MITRE ATT&CK. 
Attraverso una pipeline sperimentale rigorosa (Replication + Further Experimentation), abbiamo confrontato e ottimizzato diversi algoritmi di Machine Learning su un dataset di oltre 18.000 incidenti.
I risultati evidenziano che il modello **XGBoost** raggiunge prestazioni stato dell'arte (F1-Macro **0.9889**), mentre **KNN** emerge come soluzione ideale per l'operatività real-time, riducendo i tempi di addestramento a soli **2.9s**.

---

## 2. Metodologia e Pipeline Sperimentale

### 2.1 Design dell'Esperimento
La pipeline è stata progettata per garantire riproducibilità e robustezza rispetto ai limiti identificati nello studio di replica iniziale:
1.  **Preprocessing:** Normalizzazione semantica e *One-Hot Encoding* delle feature categoriche (Settore, Paese).
2.  **Stratificazione:** Utilizzo di `StratifiedShuffleSplit` per preservare la distribuzione delle classi rare nel Test Set (15%).
3.  **Ottimizzazione:** Transizione da `GridSearchCV` a **`RandomizedSearchCV`** per esplorare efficientemente lo spazio degli iperparametri.
4.  **Robustezza:** Integrazione di `StandardScaler` per risolvere problemi di instabilità numerica (overflow) nei modelli lineari e reti neurali.

### 2.2 Gestione delle Criticità Tecniche
Durante la sperimentazione su architetture eterogenee (macOS/Windows), sono state risolte criticità legate alla concorrenza dei processi (deadlock con LightGBM) forzando l'esecuzione sequenziale (`n_jobs=1`) nei learner interni, mantenendo il parallelismo a livello di cross-validation.

---

## 3. Analisi dei Risultati

### 3.1 Benchmark Comparativo
La Tabella 1 riporta le metriche finali ottenute sul Test Set indipendente.

| Modello | F1-Score (Macro) | Accuracy | Tempo Training (s) | Efficienza (Score/Time) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.9889 | 0.9961 | 537.0 | Bassa |
| **LightGBM** | 0.9801 | 0.9951 | 150.2 | Media |
| **SVC** | 0.9700 | 0.9903 | 46.3 | Media |
| **RandomForest** | 0.9681 | 0.9922 | 11.7 | Alta |
| **MLPClassifier** | 0.9171 | 0.9834 | 17.1 | Alta |
| **KNN** | 0.8571 | 0.9498 | 2.9 | Alta |

### 3.2 Analisi Visuale e Interpretazione
* **Curve ROC/PR (Figura 1):** L'AUC prossima a 1.0 per i modelli tree-based conferma l'eccellente capacità di ranking.
  > *[INSERIRE QUI: reports/Figure_2_ROC_PR_Comparison.png]*

* **Matrice di Confusione (Figura 2):** L'analisi degli errori mostra che le misclassificazioni sono limitate a gang correlate (es. varianti della stessa famiglia) o classi con supporto estremamente ridotto (<10 esempi).
  > *[INSERIRE QUI: reports/Figure_3_Confusion_Matrix_XGBoost.png]*

---

## 4. Discussione e Implicazioni CTI

### 4.1 Feature Importance: Le TTP come Firma
L'analisi dell'importanza delle feature conferma che il modello non apprende bias contestuali (es. Paese vittima), ma si focalizza sulle TTP tecniche. Tecniche come **T1486** (Data Encrypted) e **T1059** (Command-Line Interface) emergono come discriminanti primari.
> *[INSERIRE QUI: reports/Figure_4_Feature_Importance.png]*

### 4.2 Trade-off Accuratezza/Efficienza (RQ3)
Il grafico di trade-off evidenzia una scelta strategica per l'implementazione:
* **XGBoost:** Offre la massima precisione forense, ideale per analisi post-mortem dove il tempo di calcolo non è critico.
* **KNN:** Con un tempo di training di 2.9s, sacrifica solo lo 0.1319 di F1-score ma è **185 volte più veloce**.
  > *[INSERIRE QUI: reports/Figure_5_Performance_Tradeoff.png]*

**Raccomandazione Operativa:** Si suggerisce l'adozione di **KNN** per sistemi di *Continuous Retraining* giornalieri, al fine di mitigare il fenomeno del **Concept Drift** (evoluzione delle tattiche delle gang).

---

## 5. Conclusioni
Il progetto ha dimostrato che l'attribuzione automatizzata tramite ML è fattibile e accurata. L'ottimizzazione della pipeline ha trasformato modelli inizialmente instabili in strumenti robusti. I risultati validano l'ipotesi che le TTP costituiscano una "firma digitale" efficace per l'identificazione degli attori di minaccia.
