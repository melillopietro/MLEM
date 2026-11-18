# MLEM: Ransomware Actor-Profile Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

MLEM is a research software repository for analysing ransomware actor profiles, documented MITRE ATT&CK techniques and disclosure-level contextual information.

The repository contains two related but methodologically distinct components:

1. an aggregate closed-set actor-profile classification pipeline;
2. an RQ3 reproducibility workflow separating ATT&CK-profile similarity from chronological disclosure-context experiments.

## Research scope

### Aggregate actor-profile benchmark

The main pipeline combines:

- documented actor-level MITRE ATT&CK techniques;
- victim sector;
- victim country;
- calendar attributes derived from disclosure dates.

It supports preprocessing, stratified dataset generation, multi-model comparison, model analysis and an interactive Streamlit dashboard.

The resulting scores measure closed-set recognition within the constructed actor-profile dataset. They must not be interpreted as out-of-time attribution performance on previously unseen future disclosures.

### RQ3 reproducibility workflow

The [`rq3_validated_v2`](rq3_validated_v2/) directory contains the validation workflow used for the RQ3 experiments.

It separates:

- actor-level ATT&CK profiles, used to measure documented technical overlap;
- disclosure-level contextual observations, used for chronological classification based on victim sector and country.

The workflow includes:

- dataset and label auditing;
- conservative actor-label canonicalisation;
- ATT&CK-profile similarity analysis;
- contextual feature ablation;
- chronological validation and test splits;
- unseen-label reporting;
- rolling-origin evaluation;
- frozen-versus-updated training comparison;
- adjacent-year contextual drift analysis.

The full experimental design and interpretation boundaries are documented in [`rq3_validated_v2/README.md`](rq3_validated_v2/README.md).

## Repository structure

```text
MLEM/
├── DATASETv3.xlsx
├── Dataset Ransomware.xlsx
├── mlem_workbook.py
├── Dataset Normalized.csv
├── dashboard.py
├── normalized_dataset.py
├── dataset_ML_Formatter.py
├── generate_dataset.py
├── validate_dataset.py
├── stratification_dataset.py
├── training_manager.py
├── generate_model_analysis.py
├── generate_final_graphs.py
├── analyze_source_workbook.py
├── dataset_descriptor.py
├── signature_overlap_audit.py
├── aggregate_ablation.py
├── run_replication.py
├── ACTOR_LABEL_POLICY.md
├── experiments/configs/
├── update_mitre_db.py
├── update_cve_db.py
├── mitre_definitions.json
├── cve_definitions.json
├── rq3_validated_v2/
├── requirements.txt
├── Dockerfile
├── CITATION.cff
└── LICENSE
```

Generated datasets, trained models, local reports and temporary execution artifacts are excluded from version control.

## Data requirements

The primary workbook is **`DATASETv3.xlsx`** (sheets `Registry` and `Gang Profile`).
If it is absent, the code falls back to the legacy **`Dataset Ransomware.xlsx`**.

```text
DATASETv3.xlsx          # preferred source (v3)
Dataset Ransomware.xlsx # legacy v2 layout
Dataset Normalized.csv  # generated disclosure table
```

V3 profile fields merged into the aggregate ML dataset include:
`TTP_Techniques`, `TTP_Tactics`, origin, extortion type, programming language,
currency, operational status, lineage, confidence, law-enforcement impact,
ecosystem phase, registry flag, and documented CVE identifiers when present.

`Dataset Normalized.csv` can be regenerated from the Excel workbook with:

```bash
python normalized_dataset.py
```

The encoded machine-learning dataset and the train, validation and test partitions are generated locally and are not versioned.

## Installation (offline)

Clone the repository and use a virtual environment at the repository root. All workflows
run locally against the bundled `Dataset Ransomware.xlsx` and generated CSV files; no
external dataset download is required.

```bash
git clone https://github.com/melillopietro/MLEM.git
cd MLEM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, if `python3 -m venv` fails, install the venv package first
(for example `sudo apt install python3.12-venv`), then recreate `.venv`.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Replication bundles (reviewer-oriented)

```bash
# Paper window (through 2025-04-01)
python run_replication.py paper_jisas2026

# Full DATASETv3 through 2025-12-31
python run_replication.py extended_v3_full
```

Transparency artefacts (generated under `reports/`):

- `DATASET_DESCRIPTOR.md` / `dataset_descriptor.json`
- `signature_overlap_audit.json` / `signature_overlap_labels.csv`
- `aggregate_ablation_results.csv`

Actor-label rules: see [`ACTOR_LABEL_POLICY.md`](ACTOR_LABEL_POLICY.md).

Set a cutoff manually:

```bash
python normalized_dataset.py --as-of-date 2025-04-01
# or: export MLEM_AS_OF=2025-04-01
```

### One-shot aggregate pipeline (offline)

From the repository root, with `.venv` activated:

```bash
python normalized_dataset.py
python dataset_ML_Formatter.py
python validate_dataset.py
python generate_dataset.py
python stratification_dataset.py
python training_manager.py
python generate_model_analysis.py
python generate_final_graphs.py
python analyze_source_workbook.py
streamlit run dashboard.py
```

`analyze_source_workbook.py` writes `SOURCE_DATASET_ANALYSIS.md`, a summary of the
source Excel workbook (sheets, coverage, missing values, top gangs).

## Aggregate pipeline

Run preprocessing and dataset generation:

```bash
python normalized_dataset.py
python dataset_ML_Formatter.py
python validate_dataset.py
python generate_dataset.py
python stratification_dataset.py
```

Run the model comparison:

```bash
python training_manager.py
```

Generate local model-analysis outputs:

```bash
python generate_model_analysis.py
python generate_final_graphs.py
```

The pipeline locally generates artifacts such as:

```text
final_ml_dataset_encoded.csv
dataset_split/
model_comparison_results_final.csv
label_encoder.pkl
best_validation_model.pkl
best_validation_model.json
{Algorithm}_best_model.pkl
reports/
SOURCE_DATASET_ANALYSIS.md
```

`best_validation_model.pkl` is the classifier with the highest macro-F1 on the
validation split. The dashboard loads it before any per-algorithm pickle.
These files are excluded from version control (except `SOURCE_DATASET_ANALYSIS.md`,
which can be regenerated) because they can be rebuilt from the source workbook.

## Dashboard

Launch the Streamlit application:

```bash
streamlit run dashboard.py
```

The dashboard supports:

- execution of the aggregate pipeline;
- exploratory dataset summaries;
- model-comparison visualisations;
- actor-profile classification;
- MITRE ATT&CK lookup;
- CVE and CISA KEV lookup.

The classification interface is a research prototype. Its output is a model prediction and must not be considered definitive operational attribution.

## RQ3 validation

The RQ3 workflow is independent of aggregate model training. It reads
`Dataset Normalized.csv` and `Dataset Ransomware.xlsx` from the repository root.
Regenerate the normalized CSV first if you changed the workbook:

```bash
python normalized_dataset.py
```

Install the dedicated dependencies (once per environment):

```bash
python -m pip install -r rq3_validated_v2/requirements-rq3.txt
```

Run the complete workflow:

```bash
python rq3_validated_v2/run_all.py
```

Steps execute in order: audit, actor profiles, similarity (optional if graph extras
are missing), temporal context classification, contextual drift, report generation.
All outputs are written under `rq3_validated_v2/outputs/`.

The main report is:

```text
rq3_validated_v2/outputs/RQ3_EXPERIMENTAL_VALIDATION_REPORT.md
```

Interpretation reminder: temporal tests use victim sector and country only. Low
macro-F1 on 2024/2025 is expected; high closed-set scores from the aggregate
pipeline must not be read as out-of-time attribution.

Selected compact outputs are versioned for review. Large intermediate matrices and
prediction-level files are excluded because they can be regenerated.

## Docker

Build the container:

```bash
docker build -t mlem .
```

Run the dashboard:

```bash
docker run --rm -p 8501:8501 mlem
```

The interface will be available at:

```text
http://localhost:8501
```

## Interpretation boundaries

The following limitations are important:

- ATT&CK profiles are aggregate actor-level descriptions, not independently observed event-level technical features.
- High closed-set classification scores do not demonstrate performance on future or previously unseen disclosures.
- Context-only temporal classification based on sector and country has limited discriminatory performance.
- Similarity between actor profiles does not prove shared ownership, affiliation, rebranding or organisational continuity.
- The 2025 temporal evaluation is partial and includes records through 1 April 2025.
- Results depend on the coverage, consistency and labelling quality of the underlying disclosure and actor-profile sources.

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

This project is licensed under the [MIT License](LICENSE).
