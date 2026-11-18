"""
Aggregate (random stratified) ablation aligned with the paper feature sets.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mlem_workbook import (
    PROFILE_CATEGORICAL_COLUMNS,
    actor_key,
    clean_technique_csv,
    load_gang_profiles,
    load_registry,
    parse_cve_list,
)

from normalized_dataset import map_sector, normalize_country

OUTPUT_CSV = Path("reports/aggregate_ablation_results.csv")
MIN_SAMPLES = 10
RANDOM_STATE = 42


def build_merged_base() -> pd.DataFrame:
    reg = pd.read_csv("Dataset Normalized.csv")
    reg["gang_key"] = reg["gang"].map(actor_key)
    reg["victim_sector"] = reg["Victim sectors"].map(map_sector)
    reg["victim_country"] = reg["Victim Country"].map(normalize_country)
    reg["attack_date"] = pd.to_datetime(reg["date"], errors="coerce")
    reg = reg[reg["attack_date"].notna()].copy()
    reg["month"] = reg["attack_date"].dt.month
    reg["quarter"] = reg["attack_date"].dt.quarter

    profiles = load_gang_profiles()
    if "actor_key" in profiles.columns:
        profiles["gang_key"] = profiles["actor_key"]
    elif "gang_key" not in profiles.columns:
        profiles["gang_key"] = profiles["gang"].map(actor_key)
    prof_cols = ["gang_key", "ttp_techniques"] + [
        c for c in PROFILE_CATEGORICAL_COLUMNS if c in profiles.columns
    ]
    if "cves_exploited" in profiles.columns:
        prof_cols.append("cves_exploited")
    prof = profiles[[c for c in prof_cols if c in profiles.columns]].copy()
    if "ttp_techniques" in prof.columns:
        prof["gang_ttps"] = prof["ttp_techniques"].map(clean_technique_csv)
    else:
        prof["gang_ttps"] = ""

    merged = reg.merge(prof, on="gang_key", how="left")
    merged = merged[merged["gang_ttps"].fillna("").astype(str).str.strip().ne("")]
    counts = merged["gang"].value_counts()
    merged = merged[merged["gang"].isin(counts[counts >= MIN_SAMPLES].index)]
    return merged


def feature_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    blocks: dict[str, pd.DataFrame] = {}

    ttp = df["gang_ttps"].str.get_dummies(sep=",")
    blocks["ttp_only"] = ttp

    ctx = pd.get_dummies(
        df[["victim_sector", "victim_country"]].fillna("Unknown"),
        columns=["victim_sector", "victim_country"],
    )
    blocks["context_only"] = ctx
    blocks["ttp_plus_context"] = pd.concat([ttp, ctx], axis=1)

    cve_lists = df.get("cves_exploited", pd.Series([[]] * len(df))).map(
        lambda x: parse_cve_list(x) if not isinstance(x, list) else x
    )
    all_cves = sorted({c for lst in cve_lists for c in lst})
    if all_cves:
        cve_frame = pd.DataFrame(0, index=df.index, columns=all_cves)
        for idx, lst in cve_lists.items():
            for cve in lst:
                cve_frame.at[idx, cve] = 1
        blocks["ttp_plus_context_plus_cve"] = pd.concat([ttp, ctx, cve_frame], axis=1)
    else:
        blocks["ttp_plus_context_plus_cve"] = blocks["ttp_plus_context"].copy()

    meta_cols = [c for c in PROFILE_CATEGORICAL_COLUMNS if c in df.columns]
    if meta_cols:
        meta = pd.get_dummies(
            df[meta_cols].fillna("Unknown").astype(str),
            columns=meta_cols,
        )
        # Gang-level static metadata (like TTP): useful for audit, not OOT attribution.
        blocks["ttp_plus_context_plus_cve_plus_profile"] = pd.concat(
            [blocks["ttp_plus_context_plus_cve"], meta], axis=1
        )
    return blocks


def evaluate_block(name: str, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    le = LabelEncoder()
    y_train_e = le.fit_transform(y_train)
    y_test_e = le.transform(y_test)
    rows = []
    for model_name, model in [
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)),
    ]:
        model.fit(X_train, y_train_e)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "feature_set": name,
                "n_features": int(X.shape[1]),
                "n_labels": int(y.nunique()),
                "n_rows": int(len(X)),
                "macro_f1": float(f1_score(y_test_e, pred, average="macro")),
                "weighted_f1": float(f1_score(y_test_e, pred, average="weighted")),
                "balanced_accuracy": float(balanced_accuracy_score(y_test_e, pred)),
                "accuracy": float(accuracy_score(y_test_e, pred)),
            }
        )
    return rows


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged = build_merged_base()
    y = merged["gang"]
    blocks = feature_blocks(merged)
    results: list[dict] = []
    for name, X in blocks.items():
        results.extend(evaluate_block(name, X.astype(float), y))
    out = pd.DataFrame(results).sort_values(["feature_set", "model"])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()