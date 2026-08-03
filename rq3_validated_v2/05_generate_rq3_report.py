from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import OUTPUT_DIR


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    audit = load_json(OUTPUT_DIR / "audit_summary.json")
    split = load_json(OUTPUT_DIR / "temporal_split_summary.json")
    models = pd.read_csv(OUTPUT_DIR / "temporal_context_model_results.csv")
    ablation = pd.read_csv(OUTPUT_DIR / "temporal_context_ablation_results.csv")
    rolling = pd.read_csv(OUTPUT_DIR / "rolling_origin_context_results.csv")
    update = pd.read_csv(OUTPUT_DIR / "frozen_vs_updated_results.csv")
    drift = pd.read_csv(OUTPUT_DIR / "contextual_adjacent_year_drift.csv")

    lines = [
        "# RQ3 experimental validation report",
        "",
        "## Analytical boundary",
        "",
        "All-time ATT&CK profiles are restricted to the aggregated actor-profile baseline. "
        "They are excluded from out-of-time classification because they are not versioned "
        "by year and may contain knowledge observed after a training cutoff. Temporal "
        "classification uses only disclosure-level sector and country as its primary feature set.",
        "",
        "## Dataset audit",
        "",
    ]
    lines.extend(f"- **{key}**: {value}" for key, value in audit.items())
    lines.extend(
        [
            "",
            "## Main temporal split",
            "",
            f"- Training: **2020–2023** ({split['train_records']} eligible records)",
            f"- Validation: **2024** ({split['validation_known_records']} known; "
            f"{split['validation_unseen_records']} unseen-label records)",
            f"- Test: **2025 through {split['test_date_max']}** "
            f"({split['test_known_records']} known; {split['test_unseen_records']} unseen-label records)",
            "- The 2025 test interval is partial.",
            "",
            "| Period | Model | Coverage | Closed macro-F1 | Full macro-F1 | Closed weighted-F1 | Full accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in models.iterrows():
        lines.append(
            f"| {row['evaluation_period']} | {row['model']} | "
            f"{row['closed_set_coverage']:.4f} | {row['closed_macro_f1']:.4f} | "
            f"{row['full_macro_f1']:.4f} | {row['closed_weighted_f1']:.4f} | "
            f"{row['full_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2025 contextual ablation",
            "",
            "| Feature set | Coverage | Closed macro-F1 | Full macro-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(
            f"| {row['feature_set']} | {row['closed_set_coverage']:.4f} | "
            f"{row['closed_macro_f1']:.4f} | {row['full_macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Rolling-origin evaluation",
            "",
            "| Test year | Train through | Coverage | Closed macro-F1 | Full macro-F1 | Unseen records |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in rolling.iterrows():
        lines.append(
            f"| {int(row['test_year'])} | {int(row['train_year_max'])} | "
            f"{row['closed_set_coverage']:.4f} | {row['closed_macro_f1']:.4f} | "
            f"{row['full_macro_f1']:.4f} | {int(row['unseen_test_records'])} |"
        )

    lines.extend(
        [
            "",
            "## Frozen versus updated model on partial 2025",
            "",
            "| Policy | Coverage | Closed macro-F1 | Full macro-F1 | Δ closed macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in update.iterrows():
        lines.append(
            f"| {row['training_policy']} | {row['closed_set_coverage']:.4f} | "
            f"{row['closed_macro_f1']:.4f} | {row['full_macro_f1']:.4f} | "
            f"{row['delta_closed_macro_f1_vs_frozen']:.4f} |"
        )

    valid = drift[drift["chi_square_test_valid"]]
    lines.extend(
        [
            "",
            "## Contextual drift outputs",
            "",
            f"- Adjacent gang-year comparisons: **{len(drift)}**",
            f"- Comparisons meeting the minimum chi-square rule: **{len(valid)}**",
            "- Sector and country are reported separately with entropy, top-1/top-3 "
            "concentration, Jensen–Shannon divergence, chi-square and Cramér's V.",
            "",
            "## Interpretation boundary",
            "",
            "Closed-set metrics measure differentiation among actor labels eligible in the "
            "training interval. Full temporal metrics additionally penalise records belonging "
            "to unseen labels, which the supervised classifier cannot emit. A difference "
            "between the two therefore reflects actor turnover as well as classification error. "
            "No technical-drift claim can be derived from the all-time ATT&CK profiles.",
        ]
    )

    path = OUTPUT_DIR / "RQ3_EXPERIMENTAL_VALIDATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
