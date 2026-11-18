"""
Static TTP signature overlap audit (train/test leakage control for aggregate benchmark).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlem_workbook import actor_key, clean_technique_csv, load_gang_profiles

OUTPUT_DIR = Path("reports")
JSON_PATH = OUTPUT_DIR / "signature_overlap_audit.json"
CSV_PATH = OUTPUT_DIR / "signature_overlap_labels.csv"

MIN_TRAIN_SAMPLES = 10
RANDOM_STATE = 42


def build_labeled_frame() -> pd.DataFrame:
    if not Path("Dataset Normalized.csv").exists():
        raise FileNotFoundError("Run normalized_dataset.py first.")

    disclosures = pd.read_csv("Dataset Normalized.csv")
    disclosures = disclosures[disclosures["gang"].notna()].copy()
    disclosures["gang_key"] = disclosures["gang"].map(actor_key)

    profiles = load_gang_profiles()
    key_col = "gang_key" if "gang_key" in profiles.columns else "actor_key"
    prof = profiles.set_index(key_col)
    disclosures["ttp_signature"] = disclosures["gang_key"].map(
        lambda k: clean_technique_csv(prof.at[k, "ttp_techniques"])
        if k in prof.index
        else ""
    )
    disclosures = disclosures[disclosures["ttp_signature"].str.strip().ne("")]
    counts = disclosures["gang"].value_counts()
    keep = counts[counts >= MIN_TRAIN_SAMPLES].index
    return disclosures[disclosures["gang"].isin(keep)].copy()


def signature_lookup_predict(y_train: pd.Series, sig_train: pd.Series, sig_test: pd.Series) -> np.ndarray:
    mapping: dict[str, str] = {}
    for label, sig in zip(y_train, sig_train, strict=False):
        if sig not in mapping:
            mapping[sig] = label
        elif mapping[sig] != label:
            pass
    return np.array([mapping.get(sig, y_train.mode().iloc[0]) for sig in sig_test])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_labeled_frame()
    label_counts = df["gang"].value_counts()
    sig_by_label = (
        df.groupby("gang")["ttp_signature"].agg(lambda s: s.mode().iloc[0] if len(s) else "")
    )
    sig_to_labels = (
        sig_by_label.reset_index()
        .groupby("ttp_signature")["gang"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )
    shared = {sig: labels for sig, labels in sig_to_labels.items() if len(labels) > 1 and sig}

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.15,
        stratify=df["gang"],
        random_state=RANDOM_STATE,
    )
    train = df.loc[train_idx]
    test = df.loc[test_idx]

    train_sigs = set(train["ttp_signature"].unique())
    test_sigs = set(test["ttp_signature"].unique())
    seen_in_train = test["ttp_signature"].isin(train_sigs)

    lookup_pred = signature_lookup_predict(
        train["gang"], train["ttp_signature"], test["ttp_signature"]
    )
    lookup_macro = float(f1_score(test["gang"], lookup_pred, average="macro"))
    lookup_acc = float(accuracy_score(test["gang"], lookup_pred))

    clf_macro = None
    clf_acc = None
    if Path("model_comparison_results_final.csv").exists():
        # Reference best model metric from latest training run for comparison in the report.
        best = pd.read_csv("model_comparison_results_final.csv")["f1_macro"].max()
        clf_macro = float(best)

    payload = {
        "eligible_labels": int(label_counts.shape[0]),
        "eligible_rows": int(len(df)),
        "unique_ttp_signatures_among_labels": int(sig_by_label.nunique()),
        "labels_sharing_signatures": int(sum(len(v) for v in shared.values())),
        "signature_collisions": {k: v for k, v in list(shared.items())[:50]},
        "random_split_test_size": 0.15,
        "test_signatures_seen_in_train_fraction": float(seen_in_train.mean()),
        "exact_signature_lookup_macro_f1": lookup_macro,
        "exact_signature_lookup_accuracy": lookup_acc,
        "best_classifier_validation_macro_f1_reference": clf_macro,
        "interpretation": (
            "High lookup scores with test_signatures_seen_in_train_fraction≈1 indicate "
            "closed-set recognition of static actor profiles, not out-of-time event attribution."
        ),
    }

    label_rows = []
    for gang, sig in sig_by_label.items():
        label_rows.append(
            {
                "actor_label": gang,
                "ttp_signature": sig,
                "disclosures": int(label_counts.get(gang, 0)),
                "shared_with_labels": ",".join(
                    [g for g in sig_to_labels.get(sig, []) if g != gang]
                ),
            }
        )
    pd.DataFrame(label_rows).sort_values("disclosures", ascending=False).to_csv(
        CSV_PATH, index=False
    )
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(
        f"Labels={payload['eligible_labels']} | "
        f"signatures={payload['unique_ttp_signatures_among_labels']} | "
        f"test sig in train={payload['test_signatures_seen_in_train_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()