# RQ3 reproducibility workflow

This directory contains the validation workflow used for the RQ3 experiments in MLEM. The analysis separates two levels that should not be interpreted as the same task:

- actor-level ATT&CK profiles, used to measure documented technical overlap between ransomware labels;
- disclosure-level contextual observations, used for chronological classification experiments based on victim sector and country.

The workflow does not use all-time ATT&CK profiles as time-varying event features. This avoids treating a static actor profile as if it had been observed independently for every disclosure.

## Data requirements

The following source files must remain in the root of the MLEM repository:

```text
Dataset Normalized.csv
Dataset Ransomware.xlsx
```

The expected repository layout is:

```text
MLEM/
├── Dataset Normalized.csv
├── Dataset Ransomware.xlsx
└── rq3_validated_v2/
    ├── README.md
    ├── requirements-rq3.txt
    ├── common.py
    ├── run_all.py
    ├── 01_audit_dataset.py
    ├── 02_build_actor_profiles.py
    ├── 03_actor_similarity.py
    ├── 04_temporal_context_classification.py
    ├── 05_generate_rq3_report.py
    ├── 06_contextual_drift.py
    └── outputs/
```

## Experimental design

### Actor-profile analysis

ATT&CK techniques are extracted from the `Ransomware Gang Profile` worksheet. Labels are canonicalised conservatively: only differences in case, whitespace and punctuation are merged. Rebrands, successors and versioned labels remain distinct.

The workflow produces actor-technique profiles, pairwise Jaccard and cosine similarities, agglomerative clusters and network visualisations. Similarity is evidence of overlap in the documented profiles; it is not evidence of shared ownership, affiliation or organisational continuity.

### Temporal context analysis

The chronological evaluation uses disclosure-level attributes that are observable for each record:

- victim sector;
- victim country.

Calendar attributes are retained only as a sensitivity analysis.

The main split is:

- training: 2020–2023;
- validation: 2024;
- test: 2025, through 1 April 2025;
- minimum training support: 10 disclosures per label;
- random seed: 42.

Labels absent from the training period are reported separately as unseen. Closed-set and full temporal-test metrics are both retained in the output files.

The workflow also includes:

- contextual feature ablation;
- rolling-origin evaluation;
- frozen-versus-updated training comparison;
- adjacent-year contextual drift analysis.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r rq3_validated_v2/requirements-rq3.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the workflow

From the repository root:

```bash
python rq3_validated_v2/run_all.py
```

All generated files are written to:

```text
rq3_validated_v2/outputs/
```

The main summary is:

```text
rq3_validated_v2/outputs/RQ3_EXPERIMENTAL_VALIDATION_REPORT.md
```

## Main outputs

- `audit_summary.json`: dataset boundary, duplicate removal and label audit;
- `actor_profile_metadata.csv`: actor-profile coverage and technique counts;
- `actor_top_similarity_pairs.csv`: highest documented ATT&CK-profile similarities;
- `temporal_split_summary.json`: chronological split and unseen-label counts;
- `temporal_context_model_results.csv`: model selection and isolated test results;
- `temporal_context_ablation_results.csv`: sector/country feature comparison;
- `rolling_origin_context_results.csv`: repeated historical-to-future evaluation;
- `frozen_vs_updated_results.csv`: effect of updating the training window;
- `contextual_adjacent_year_drift.csv`: adjacent-year distribution change;
- `RQ3_EXPERIMENTAL_VALIDATION_REPORT.md`: compact report of the complete workflow.

## Interpretation

The original high-scoring MLEM classifier is a closed-set recognition experiment built from actor-level profiles. It should not be presented as an out-of-time event-attribution result. The experiments in this directory provide a separate assessment of technical-profile overlap and temporal contextual differentiation.
