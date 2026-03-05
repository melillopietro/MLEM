# 🛡️ MLEM: Ransomware Attribution Framework
### Advanced Hybrid Profiling & Forensic Attribution System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Overview

**MLEM (Machine Learning for Enterprise Malware)** is an automated forensic attribution framework designed to identify ransomware threat actors based on their operational behavior.

Traditional detection systems rely on static signatures (hashes, IPs, domains) which are easily rotated by attackers. MLEM shifts the focus to **Behavioral Profiling** by analyzing **TTPs (Tactics, Techniques, and Procedures)** derived from the **MITRE ATT&CK Enterprise Matrix**. This allows for high-confidence attribution even when attackers modify their infrastructure, payloads, or operational artifacts.

The framework is built upon a massive dataset of **18,600+ real-world ransomware incidents (2020–2025)** and utilizes multiple machine learning architectures to provide both predictive power and forensic explainability.

---

## 🎯 Key Capabilities

* **🔬 Multi-Architecture ML Benchmarking**: Automated training and comparison of five distinct architectures (XGBoost, Random Forest, SVM, KNN, MLP).
* **📊 Forensic Intelligence Dashboard**: Interactive **Streamlit** interface featuring Sankey diagrams for attack flows, geographic victim distribution, and MITRE TTP heatmaps.

* **🧠 Model Explainability (SHAP)**: Integration of **SHapley Additive Explanations** to provide "Glass Box" AI, explaining exactly which TTPs influenced each attribution.
* **🕸️ RaaS Infrastructure Detection**: Graph-based similarity analysis to detect **Ransomware-as-a-Service** patterns and link "copycat" groups to known affiliates.
* **📑 Automated Scientific Reporting**: Generates technical reports with precision/recall metrics and confusion matrices suitable for incident response and academic use.
* **🔌 Offline MITRE Knowledge Base**: Local database of techniques, tactics, and CVE relationships for air-gapped forensic analysis.

---

## 📊 Performance Benchmarks (2025 Dataset)

The system handles the inherent class imbalance of the ransomware ecosystem by optimizing for **Macro F1-Score**.

| Model Architecture | Accuracy | F1-Score (Macro) | Training Time | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Champion)** | **99.56%** | **0.9882** | 78.27s | 🏆 State-of-the-Art |
| **Random Forest** | 99.27% | 0.9706 | 2.69s | High Efficiency |
| **SVM (RBF Kernel)** | 99.03% | 0.9700 | 19.72s | Robust |
| **Neural Network (MLP)** | 98.73% | 0.9506 | 28.99s | Deep Learning |
| **LightGBM** | 31.56% | 0.2214 | 10.72s | Underfitting |



---

## 🧠 Dataset & Methodology

The framework leverages a structured intelligence dataset including:

* **18,600+** Unique Ransomware Incidents.
* **200+** Threat Actor Labels (e.g., LockBit, BlackBasta, Akira, CL0P).
* **Features**: Victim sector, victim country, CVE exploitation, and 150+ MITRE ATT&CK techniques.

> **Note**: Macro F1-score is prioritized over raw accuracy to ensure reliable classification of emerging or minority ransomware gangs, which are typically underrepresented in cyber threat intelligence (CTI) feeds.

---

## 🏗 Project Structure

```text
mlem/
├── data/                   # Raw and processed incident datasets
├── models/                 # Serialized model checkpoints (.pkl / .onnx)
├── notebooks/              # Research, EDA, and feature engineering logs
├── src/                    # Core Framework Logic
│   ├── preprocessing/      # TTP encoding and data normalization
│   ├── training/           # Training pipelines & hyperparameter tuning
│   ├── attribution/        # Inference engine for live forensic cases
│   └── explainability/     # SHAP and LIME integration modules
├── dashboard/              # Streamlit UI source code (app.py)
├── reports/                # Automatically generated PDF/JSON forensic reports
├── requirements.txt        # Project dependencies
└── main.py                 # CLI Entry point for training/inference
```

## ⚙️ Installation & Setup
Requirements
Python 3.12+

16 GB RAM recommended for processing the full 2020–2025 dataset.

## 1. Clone & Environment
```text
git clone https://github.com/your-username/mlem.git
cd mlem

# Create Virtual Environment
python -m venv venv

# Activate Environment
# On Linux/macOS:
source venv/bin/activate  
# On Windows:
.\venv\Scripts\activate
```

## 2. Install Dependencies
```text
pip install -r requirements.txt
```

## 3. Launch the Intelligence Dashboard
```text
streamlit run dashboard/app.py
```
## 📜 License
This project is licensed under the MIT License.