from __future__ import annotations

import pandas as pd

from common import (
    DATASET_NORMALIZED,
    MIN_TRAIN_SAMPLES,
    OUTPUT_DIR,
    build_actor_registry,
    load_actor_profiles,
    load_disclosures,
    write_json,
)


def main() -> None:
    raw = pd.read_csv(DATASET_NORMALIZED)
    parsed_raw_labels = load_disclosures(deduplicate=False, canonicalize_labels=False)
    dedup_raw_labels = load_disclosures(deduplicate=True, canonicalize_labels=False)
    parsed = load_disclosures(deduplicate=False, canonicalize_labels=True)
    dedup = load_disclosures(deduplicate=True, canonicalize_labels=True)
    actor_sets, invalid_tokens, tactic_tokens = load_actor_profiles()
    _, mapping = build_actor_registry()

    mapping.to_csv(OUTPUT_DIR / "actor_label_mapping.csv", index=False)

    counts = dedup["gang"].value_counts()
    eligible_labels = set(counts[counts >= MIN_TRAIN_SAMPLES].index)
    labels_with_valid_profiles = eligible_labels.intersection(actor_sets)
    labels_without_valid_profiles = sorted(eligible_labels.difference(actor_sets))

    year_counts = (
        dedup.groupby("year")
        .agg(disclosures=("gang", "size"), actor_labels=("gang", "nunique"))
        .reset_index()
    )
    year_counts.to_csv(OUTPUT_DIR / "audit_year_counts.csv", index=False)

    label_counts = counts.rename_axis("actor_label").reset_index(name="disclosures")
    label_counts["eligible_min_10"] = label_counts["disclosures"] >= MIN_TRAIN_SAMPLES
    label_counts["has_valid_attack_profile"] = label_counts["actor_label"].isin(actor_sets)
    label_counts.to_csv(OUTPUT_DIR / "audit_actor_counts.csv", index=False)

    invalid_tokens.to_csv(OUTPUT_DIR / "invalid_technical_tokens.csv", index=False)
    tactic_tokens.to_csv(OUTPUT_DIR / "excluded_tactic_tokens.csv", index=False)
    pd.DataFrame({"actor_label": labels_without_valid_profiles}).to_csv(
        OUTPUT_DIR / "eligible_labels_without_valid_attack_profile.csv", index=False
    )

    summary = {
        "raw_rows": int(len(raw)),
        "rows_with_valid_label_and_date": int(len(parsed)),
        "deduplicated_rows": int(len(dedup)),
        "exact_duplicate_rows_removed_before_label_normalisation": int(len(parsed_raw_labels) - len(dedup_raw_labels)),
        "additional_duplicate_rows_removed_after_label_normalisation": int(len(dedup_raw_labels) - len(dedup)),
        "raw_actor_labels": int(parsed_raw_labels["gang"].nunique()),
        "canonical_actor_labels": int(dedup["gang"].nunique()),
        "lexical_label_variants_remapped": int(mapping["lexical_variant_merged"].sum()),
        "eligible_labels_min_10_disclosures": int(len(eligible_labels)),
        "eligible_labels_with_valid_attack_profile": int(len(labels_with_valid_profiles)),
        "eligible_labels_without_valid_attack_profile": labels_without_valid_profiles,
        "invalid_technical_token_rows": int(len(invalid_tokens)),
        "excluded_tactic_token_rows": int(len(tactic_tokens)),
        "date_min": str(dedup["date"].min().date()),
        "date_max": str(dedup["date"].max().date()),
    }
    write_json(OUTPUT_DIR / "audit_summary.json", summary)

    print("Dataset audit completed")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
