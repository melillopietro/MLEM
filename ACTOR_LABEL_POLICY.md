# Actor label policy (MLEM)

This document defines how ransomware actor labels are handled in the disclosure
layer and in the machine-learning pipelines. It supports replication and
addresses reviewer questions on naming, rebranding, and versioned labels.

## Principles

1. **Actor labels are operational identifiers**, not proof of a single criminal
   organization or legal attribution.
2. **Rebrands and successors stay distinct** unless they differ only by case,
   spacing, or punctuation (conservative lexical merge).
3. **LockBit versioning** (`LockBit`, `lockbit2`, `lockbit3`, …) remains
   **separate labels** in ML and RQ3 unless a study explicitly defines a
   brand-level aggregation rule.
4. **Matching to profiles** uses `actor_key` (lowercase alphanumeric fold), not
   exact string equality, to tolerate vendor spelling variants.

## `actor_key` rule

```text
actor_key(label) = lowercase alphanumeric collapse of the label string
```

Example: `LockBit 3.0` and `lockbit3` may map to the same key and the same
profile row when the gang profile sheet uses one canonical spelling.

## Canonicalisation (RQ3)

Implemented in `rq3_validated_v2/common.py` → `build_actor_registry()`:

- Merge only **lexical variants** of the same operational label.
- Do **not** merge distinct public names (e.g. Conti vs Monti) even when ATT&CK
  profiles overlap.
- Output: `rq3_validated_v2/outputs/actor_label_mapping.csv`.

## LockBit and versioned labels

| Use case | Policy |
|----------|--------|
| RQ1 narrative case study | Treat **LockBit** as one evolutionary line in gray literature. |
| Figure 1 / ML / RQ3 counts | Keep **lockbit2**, **lockbit3**, **LockBit** as separate labels when they appear in the registry. |
| Paper text | State explicitly which convention each figure/table uses. |

## Missing or ambiguous labels

- Rows without a parseable date are excluded from temporal experiments.
- Rows without a non-empty ATT&CK technique profile are excluded from the
  **aggregate** ML encoder (`dataset_ML_Formatter.py`) but may remain in
  disclosure-only analyses.
- `Unknown` gang labels are retained in the registry; they are rarely eligible
  for supervised classification (minimum support threshold).

## Temporal cutoff (`as-of`)

- Environment variable: `MLEM_AS_OF=YYYY-MM-DD`
- Config file: `mlem_run_config.json`
- CLI: `python normalized_dataset.py --as-of-date 2025-04-01`
- Paper replication bundle uses **2025-04-01** (partial 2025 interval).

## Duplicates

- RQ3 removes exact duplicates on `(victim, gang, date, sector, country)` after
  canonicalisation (see `audit_summary.json`).
- The normalized CSV may still contain legacy duplicate rows until deduplication
  is applied in downstream workflows.