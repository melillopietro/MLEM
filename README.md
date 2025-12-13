# MLEM: Hybrid Ransomware Attribution Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20ScikitLearn-green)
![License](https://img.shields.io/badge/License-MIT-grey)


**MLEM** is an advanced Cyber Threat Intelligence (CTI) framework designed to attribute Ransomware-as-a-Service (RaaS) attacks to specific threat groups. By leveraging a **Hybrid Profiling** approach, the system combines technical behavioral signatures (MITRE ATT&CK TTPs) with contextual victimology data (Industrial Sector and Geography) to achieve high-precision forensic attribution.

The framework utilizes **XGBoost** for classification on highly sparse datasets and integrates **SHAP (SHapley Additive exPlanations)** to provide granular, interpretable evidence for every attribution decision, eliminating the "black box" problem in AI-driven forensics.

---

## Key Features

* **Hybrid Profiling Engine:** Integrates technical artifacts (TTPs) with victimology metadata. Statistical ablation studies demonstrate that this hybrid approach improves attribution performance by **+5.24%** compared to purely technical baselines.
* **State-of-the-Art Classification:** Built on XGBoost (eXtreme Gradient Boosting), optimized for sparse matrices and imbalanced multi-class datasets (handling over 50 distinct ransomware families).
* **Explainable AI (XAI):**
    * **Global Explainability:** Identifies top discriminative features across the entire threat landscape.
    * **Local Forensics:** Provides instance-level SHAP waterfall charts to justify specific attribution decisions.
* **Geospatial Intelligence:** Interactive 3D visualization mapping global victim distribution against active threat actors.
* **Automated Scientific Validation:** Includes a suite of scripts for Stratified K-Fold Cross-Validation, Ablation Studies, and generation of engineering-grade reports (including MCC, Kappa, and Sparsity analysis).

---

## Technical Architecture

The pipeline consists of three main stages:

1.  **Preprocessing & Vectorization:** Converts raw CTI reports into a normalized feature vector space. Handles categorical encoding for victim sectors and countries.
2.  **Model Training & Optimization:** Trains an ensemble of decision trees using Gradient Boosting. Hyperparameters are tuned to minimize log-loss while maximizing the Macro F1-Score.
3.  **Forensic Dashboard:** A Streamlit-based interface for analysts to interact with the model, simulate attacks, and visualize intelligence data.

---

## Installation

### Prerequisites
* Python 3.10 or higher (Python 3.12 recommended)
* pip package manager

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/mlem-attribution.git](https://github.com/your-username/mlem-attribution.git)
    cd mlem-attribution
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Launching the Dashboard
The primary interface is the web-based dashboard. To start the system:

```bash
python -m streamlit run dashboard.py
