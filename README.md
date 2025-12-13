# 🛡️ MLEM: Ransomware Attribution Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20ScikitLearn-green)
![License](https://img.shields.io/badge/License-MIT-grey)

> **Advanced Cyber Threat Intelligence Tool** per l'attribuzione forense di gang Ransomware utilizzando Feature Ibride (TTPs MITRE ATT&CK + Vittimologia) e algoritmi di Machine Learning allo stato dell'arte.

---

## 📋 Overview

**MLEM** (Machine Learning for Enterprise Malware) è un framework full-stack che automatizza il processo di attribuzione degli attacchi ransomware. A differenza dei sistemi tradizionali basati solo su hash o firme statiche, MLEM analizza il **comportamento** (Tattiche, Tecniche e Procedure) e il **contesto** (Settore e Paese della vittima) per identificare l'attore criminale (RaaS Gang).

Il progetto include una **Dashboard Interattiva** per analisti SOC che permette di:
1.  Addestrare e confrontare 5 famiglie di modelli ML.
2.  Simulare scenari di attacco reali.
3.  Eseguire attribuzioni forensi con gestione dell'incertezza.

---

## ✨ Key Features

### 🧠 1. Suite Multi-Modello
Il sistema non si limita a un solo algoritmo, ma addestra e confronta in tempo reale una suite completa per garantire la robustezza scientifica:
* **Gradient Boosting:** XGBoost (Best Performer), LightGBM.
* **Ensemble:** Random Forest.
* **Kernel Methods:** Support Vector Machines (SVM).
* **Neural Networks:** Multi-Layer Perceptron (MLP).
* **Instance-Based:** K-Nearest Neighbors (KNN).

### 🕵️ 2. Forensic Investigator "Smart"
Un'interfaccia dedicata agli analisti con funzionalità avanzate:
* **🧪 Real-World Profiling:** Caricamento automatico di "fingerprint" reali dal database storico (es. *Carica profilo LockBit3*) per validare il modello.
* **⚠️ Logica Anti-Allucinazione (N/D):** Sistema di sicurezza che restituisce **"N/D (Analisi Inconclusiva)"** se i dati di input sono scarsi (es. 1 sola TTP) o la confidenza del modello è < 50%, riducendo drasticamente i falsi positivi.

### 🧬 3. Feature Engineering Ibrido
Utilizza un vettore di feature complesso che combina:
* **Tecniche MITRE ATT&CK** (es. *T1486 - Data Encrypted*).
* **Vittimologia Geopolitica** (es. *Victim Country*).
* **Vittimologia Industriale** (es. *Victim Sector*).

---

## 📊 Performance

I risultati sperimentali sul Test Set dimostrano prestazioni allo stato dell'arte:

| Modello | F1-Score (Macro) | Accuratezza | Note |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **0.9882** | **99.56%** | 🏆 Modello utilizzato in produzione |
| **SVM** | 0.9700 | 99.02% | Eccellente generalizzazione |
| **Random Forest** | 0.9598 | 99.07% | Alta efficienza computazionale |
| **NeuralNet** | 0.9506 | 98.73% | Valida la separabilità dei dati |

---

## 🚀 Installation & Setup

### Prerequisiti
* Python 3.10 o superiore.

### 1. Clona la repository
```bash
git clone [https://github.com/tuo-username/MLEM-Ransomware-Attribution.git](https://github.com/tuo-username/MLEM-Ransomware-Attribution.git)
cd MLEM-Ransomware-Attribution