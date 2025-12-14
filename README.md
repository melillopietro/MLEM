# 🛡️ MLEM: Ransomware Attribution Framework
### Advanced Hybrid Profiling & Forensic Attribution System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Overview
**MLEM (Machine Learning for Enterprise Malware)** is an automated forensic framework designed to attribute ransomware attacks to specific Threat Actors (Gangs) based on their **TTPs (Tactics, Techniques, and Procedures)**.

Leveraging a dataset of over **18,600 real-world incidents** (2020-2025), the system utilizes **XGBoost, SVM, and Deep Learning** to map attack patterns against the **MITRE ATT&CK Enterprise Matrix**. Unlike traditional signature-based detection, MLEM analyzes behavioral vectors to identify actors even when infrastructure changes.

### 🎯 Key Capabilities
* **Multi-Architecture Benchmarking:** Automated training & comparison of XGBoost, Random Forest, SVM, KNN, and MLP Neural Networks.
* **Forensic Intelligence Dashboard:** Real-time visualization of attack flows (Sankey), geographic density, and TTP heatmaps.
* **RaaS Detection Engine:** Graph Theory analysis to identify "Copycat Gangs" (100% similarity) indicating shared Ransomware-as-a-Service infrastructure.
* **Granular Explainability:** Local **SHAP (SHapley Additive exPlanations)** analysis to explain *why* a specific attribution was made.
* **Automated Reporting V3:** Generates university-grade PDF technical reports with per-class precision/recall metrics.
* **Offline Knowledge Base:** Integrated search engine for MITRE ATT&CK definitions.

---

## 📊 Performance Benchmarks (2025 Dataset)
The framework has been tested on a proprietary dataset exhibiting high class imbalance (Power-Law distribution).

| Model Architecture | Accuracy (Global) | F1-Score (Macro) | Training Time | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost (Champion)** | **99.56%** | **0.9882** | 78.27s | 🏆 **SOTA** |
| Random Forest | 99.27% | 0.9706 | 2.69s | Efficient |
| SVM (Kernel RBF) | 99.03% | 0.9700 | 19.72s | Robust |
| Neural Net (MLP) | 98.73% | 0.9506 | 28.99s | Good |
| LightGBM | 31.56% | 0.2214 | 10.72s | Underfitting |

> **Note:** F1-Macro was prioritized over Accuracy to ensure correct classification of emerging/minority ransomware gangs.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher (3.12 recommended)
* 64-bit Architecture (for large dataset processing)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-repo/mlem-framework.git](https://github.com/your-repo/mlem-framework.git)
cd mlem-framework