#!/usr/bin/env python3
"""Flatten experiments/comparison/summary.json into paper-ready CSV tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "experiments" / "comparison"
SUMMARY = COMP / "summary.json"
OUT_DIR = COMP / "exports"

SCENARIO_LABELS = {
    "legacy_paper": "dataset_originale",
    "v3_paper": "v3_finestra_paper",
    "v3_full": "v3_23807_vittime",
}

PAPER_RF = {
    "ttp_only": 0.9247,
    "ttp_plus_context": 0.9726,
    "context_only": 0.1892,
}


def load_summary() -> list[dict]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))["scenarios"]


def export_corpus(scenarios: list[dict]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        c = s["corpus"]
        rows.append(
            {
                "dataset": SCENARIO_LABELS[s["scenario_id"]],
                "workbook": c["workbook"],
                "as_of_date": c.get("as_of_date") or "nessuno",
                "righe_excel_registry": c["registry_sheet_rows_excel"],
                "righe_dopo_filtro_temporale": c["registry_rows_with_as_of"],
                "gang_distinte": c["distinct_gangs_as_of"],
                "profili_gang": c["profile_rows"],
                "profili_con_tecniche": c["profiles_with_techniques"],
                "profili_con_cve": c["profiles_with_cve"],
                "data_min": c["date_min"],
                "data_max_file": c["date_max_raw"],
                "data_max_usato": c["date_max_as_of"],
                "disclosure_2025_nel_file": c["disclosures_by_year_raw"].get(
                    "2025", c["disclosures_by_year_raw"].get("2025.0")
                ),
            }
        )
    return pd.DataFrame(rows)


def export_ablation(scenarios: list[dict]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s in scenarios:
        sid = s["scenario_id"]
        path = COMP / sid / "aggregate_ablation_results.csv"
        df = pd.read_csv(path)
        df.insert(0, "dataset", SCENARIO_LABELS[sid])
        df["paper_rf_macro_f1_ref"] = df["feature_set"].map(
            lambda k: PAPER_RF.get(k, None)
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def export_signature(scenarios: list[dict]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        sig = s["signature_audit"]
        rows.append(
            {
                "dataset": SCENARIO_LABELS[s["scenario_id"]],
                "righe_eleggibili": sig["eligible_rows"],
                "label_eleggibili": sig["eligible_labels"],
                "firme_ttp_uniche": sig["unique_ttp_signatures"],
                "frazione_firme_test_in_train": sig[
                    "test_signatures_seen_in_train_fraction"
                ],
                "lookup_firma_macro_f1": sig["exact_signature_lookup_macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def export_rq3(scenarios: list[dict]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        audit = s["rq3"].get("audit", {})
        for t in s["rq3"].get("temporal_best", []):
            rows.append(
                {
                    "dataset": SCENARIO_LABELS[s["scenario_id"]],
                    "rq3_raw_rows": audit.get("raw_rows"),
                    "rq3_deduplicated_rows": audit.get("deduplicated_rows"),
                    "rq3_canonical_labels": audit.get("canonical_actor_labels"),
                    "rq3_labels_min_10": audit.get("eligible_labels_min_10_disclosures"),
                    "rq3_labels_con_profilo_attck": audit.get(
                        "eligible_labels_with_valid_attack_profile"
                    ),
                    "periodo": t["evaluation_period"],
                    "modello": t["model"],
                    "closed_macro_f1": t["closed_macro_f1"],
                    "full_macro_f1": t["full_macro_f1"],
                    "copertura_closed_set": t["closed_set_coverage"],
                }
            )
    return pd.DataFrame(rows)


def export_tab5_rf_wide(scenarios: list[dict]) -> pd.DataFrame:
    """One row per dataset: RF macro-F1 for paper-like feature sets."""
    rows = []
    for s in scenarios:
        abl = s["ablation_rf_macro_f1"]
        row = {
            "dataset": SCENARIO_LABELS[s["scenario_id"]],
            "n_righe_ml": int(abl.get("_n_rows", 0)),
            "n_label_ge_10": int(abl.get("_n_labels", 0)),
            "paper_ttp_only": PAPER_RF["ttp_only"],
            "rf_ttp_only": abl.get("ttp_only"),
            "paper_ttp_plus_context": PAPER_RF["ttp_plus_context"],
            "rf_ttp_plus_context": abl.get("ttp_plus_context"),
            "rf_ttp_plus_context_plus_cve": abl.get("ttp_plus_context_plus_cve"),
            "paper_context_only": PAPER_RF["context_only"],
            "rf_context_only": abl.get("context_only"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(
            f"Missing {SUMMARY}. Run experiments/compare_three_datasets.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_summary()
    export_corpus(scenarios).to_csv(OUT_DIR / "metriche_corpus.csv", index=False)
    export_ablation(scenarios).to_csv(OUT_DIR / "metriche_ablation_completa.csv", index=False)
    export_tab5_rf_wide(scenarios).to_csv(OUT_DIR / "metriche_tab5_rf.csv", index=False)
    export_signature(scenarios).to_csv(OUT_DIR / "metriche_signature_audit.csv", index=False)
    export_rq3(scenarios).to_csv(OUT_DIR / "metriche_rq3_cronologico.csv", index=False)
    print(f"Wrote CSV exports under {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()