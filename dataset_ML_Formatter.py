import pandas as pd

from mlem_workbook import (
    PROFILE_CATEGORICAL_COLUMNS,
    actor_key,
    clean_technique_csv,
    load_gang_profiles,
    parse_cve_list,
    resolve_workbook,
    workbook_summary,
)

'''
Author: Pietro Melillo
Description: Builds the encoded ML dataset from normalized disclosures and gang profiles.
'''

standardized_file = "Dataset Normalized.csv"
output_file = "final_ml_dataset_encoded.csv"

df_std = pd.read_csv(standardized_file)
df_std["date"] = pd.to_datetime(df_std["date"], errors="coerce")
df_std["gang_key"] = df_std["gang"].map(actor_key)

df_profile = load_gang_profiles()
df_profile["gang_key"] = df_profile["actor_key"]

ttp_fields = ["ttp_techniques", "ttp_tactics"] + PROFILE_CATEGORICAL_COLUMNS
if "cves_exploited" in df_profile.columns:
    ttp_fields.append("cves_exploited")

profile_cols = ["gang_key", "gang"] + [c for c in ttp_fields if c in df_profile.columns]
df_profile = df_profile[profile_cols].rename(columns={"gang": "profile_gang_label"})

df_profile["gang_ttps"] = df_profile.get("ttp_techniques", "").map(clean_technique_csv)
if "ttp_tactics" in df_profile.columns:
    from mlem_workbook import clean_tactic_csv

    df_profile["gang_tactics"] = df_profile["ttp_tactics"].map(clean_tactic_csv)
else:
    df_profile["gang_tactics"] = ""

df_profile["gang_ttps"] = df_profile.apply(
    lambda row: ",".join(
        p for p in (row["gang_ttps"], row.get("gang_tactics", "")) if p
    ),
    axis=1,
)

if "cves_exploited" in df_profile.columns:
    df_profile["parsed_cves"] = df_profile["cves_exploited"].map(parse_cve_list)
else:
    df_profile["parsed_cves"] = [[] for _ in range(len(df_profile))]

df_merged = df_std.merge(df_profile, on="gang_key", how="left")
df_merged = df_merged[
    df_merged["gang_ttps"].notna() & (df_merged["gang_ttps"].astype(str).str.strip() != "")
]

df_final = df_merged[
    [
        "gang",
        "gang_ttps",
        "date",
        "Victim sectors",
        "Victim Country",
    ]
    + [c for c in PROFILE_CATEGORICAL_COLUMNS if c in df_merged.columns]
].rename(
    columns={
        "gang": "label_gang",
        "date": "attack_date",
        "Victim sectors": "victim_sector",
        "Victim Country": "victim_country",
    }
)

df_final["attack_date"] = pd.to_datetime(df_final["attack_date"], errors="coerce")
df_final = df_final[df_final["attack_date"].notna()].copy()
df_final["year"] = df_final["attack_date"].dt.year
df_final["month"] = df_final["attack_date"].dt.month
df_final["dayofweek"] = df_final["attack_date"].dt.day_of_week

cat_cols = ["victim_sector", "victim_country"] + [
    c for c in PROFILE_CATEGORICAL_COLUMNS if c in df_final.columns
]
for col in cat_cols:
    df_final[col] = df_final[col].fillna("Unknown").astype(str).str.strip()
    df_final.loc[df_final[col] == "", col] = "Unknown"

df_encoded = pd.get_dummies(df_final, columns=cat_cols)
df_ttp = df_final["gang_ttps"].str.get_dummies(sep=",")

# Documented CVE tokens from gang profile (not expanded victim-level CVE observations).
if "parsed_cves" in df_merged.columns:
    cve_series = df_merged.loc[df_final.index, "parsed_cves"]
    all_cves = sorted({cve for lst in cve_series for cve in lst})
    if all_cves:
        cve_frame = pd.DataFrame(0, index=df_final.index, columns=all_cves)
        for idx, lst in cve_series.items():
            for cve in lst:
                cve_frame.at[idx, cve] = 1
        df_encoded = pd.concat([df_encoded, cve_frame], axis=1)

df_encoded = pd.concat(
    [df_encoded.drop(columns=["gang_ttps", "attack_date"], errors="ignore"), df_ttp],
    axis=1,
)

bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

summary = workbook_summary()
df_encoded.to_csv(output_file, index=False)
print(f"Dataset saved: {output_file}")
print(
    f"   Rows: {len(df_encoded)} | Features: {df_encoded.shape[1]} | "
    f"Source workbook: {summary['path']} ({summary['schema']})"
)