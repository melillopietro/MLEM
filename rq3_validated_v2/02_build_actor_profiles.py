from __future__ import annotations

from collections import Counter

import pandas as pd

from common import MIN_TRAIN_SAMPLES, OUTPUT_DIR, load_actor_profiles, load_disclosures


def main() -> None:
    disclosures = load_disclosures(deduplicate=True, canonicalize_labels=True)
    actor_sets, invalid_tokens, tactic_tokens = load_actor_profiles()

    counts = disclosures["gang"].value_counts()
    eligible = sorted(
        actor
        for actor, count in counts.items()
        if count >= MIN_TRAIN_SAMPLES and actor in actor_sets
    )

    all_techniques = sorted({t for actor in eligible for t in actor_sets[actor]})
    matrix_rows = []
    long_rows = []
    metadata_rows = []

    for actor in eligible:
        actor_df = disclosures[disclosures["gang"] == actor]
        techniques = actor_sets[actor]
        row = {"actor_label": actor}
        row.update({technique: int(technique in techniques) for technique in all_techniques})
        matrix_rows.append(row)

        for technique in sorted(techniques):
            long_rows.append({"actor_label": actor, "technique_id": technique})

        metadata_rows.append(
            {
                "actor_label": actor,
                "disclosures": int(len(actor_df)),
                "first_disclosure": actor_df["date"].min().date(),
                "last_disclosure": actor_df["date"].max().date(),
                "active_years_observed": int(actor_df["year"].nunique()),
                "documented_attack_techniques": int(len(techniques)),
            }
        )

    matrix = pd.DataFrame(matrix_rows).sort_values("actor_label")
    long_df = pd.DataFrame(long_rows).sort_values(["actor_label", "technique_id"])
    metadata = pd.DataFrame(metadata_rows).sort_values("actor_label")

    prevalence = Counter()
    for actor in eligible:
        prevalence.update(actor_sets[actor])
    prevalence_df = pd.DataFrame(
        [
            {
                "technique_id": technique,
                "actor_labels": count,
                "actor_prevalence": count / len(eligible) if eligible else 0.0,
            }
            for technique, count in prevalence.most_common()
        ]
    )

    matrix.to_csv(OUTPUT_DIR / "actor_profile_matrix.csv", index=False)
    long_df.to_csv(OUTPUT_DIR / "actor_profiles_long.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "actor_profile_metadata.csv", index=False)
    prevalence_df.to_csv(OUTPUT_DIR / "technique_prevalence.csv", index=False)
    invalid_tokens.to_csv(OUTPUT_DIR / "invalid_technical_tokens.csv", index=False)
    tactic_tokens.to_csv(OUTPUT_DIR / "excluded_tactic_tokens.csv", index=False)

    print(f"Actor-profile dataset created: {len(matrix)} labels x {len(all_techniques)} ATT&CK techniques")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
