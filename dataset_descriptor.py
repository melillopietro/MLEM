"""
Generate paper-ready dataset transparency reports (descriptor tables + JSON summary).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlem_config import get_as_of_date
from mlem_workbook import (
    actor_key,
    clean_technique_csv,
    load_gang_profiles,
    load_registry,
    parse_cve_list,
    resolve_workbook,
    workbook_summary,
)

OUTPUT_DIR = Path("reports")
MD_PATH = OUTPUT_DIR / "DATASET_DESCRIPTOR.md"
JSON_PATH = OUTPUT_DIR / "dataset_descriptor.json"


def describe_registry(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["date"], errors="coerce")
    year_counts = dates.dt.year.value_counts().sort_index()
    out = {
        "rows": int(len(df)),
        "distinct_gangs": int(df["gang"].nunique()),
        "date_min": str(dates.min().date()) if dates.notna().any() else None,
        "date_max": str(dates.max().date()) if dates.notna().any() else None,
        "missing_date": int(dates.isna().sum()),
        "duplicate_rows_exact": int(df.duplicated().sum()),
        "missing_country": int(df["Victim Country"].isna().sum()) if "Victim Country" in df.columns else None,
        "missing_sector": int((df["Victim sectors"].isna() | (df["Victim sectors"] == "")).sum())
        if "Victim sectors" in df.columns
        else None,
        "unknown_sector": int((df["Victim sectors"].astype(str).str.strip().str.lower() == "unknown").sum())
        if "Victim sectors" in df.columns
        else None,
        "disclosures_by_year": {str(int(k)): int(v) for k, v in year_counts.items()},
        "top_gangs_15": df["gang"].value_counts().head(15).astype(int).to_dict(),
    }
    return out


def describe_profiles(profiles: pd.DataFrame) -> dict:
    tech = profiles["ttp_techniques"].fillna("").astype(str).map(clean_technique_csv)
    has_tech = tech.str.strip().ne("")
    cve_counts = profiles.get("cves_exploited", pd.Series(dtype=object)).map(parse_cve_list)
    with_cve = cve_counts.map(len).gt(0) if len(cve_counts) else pd.Series(dtype=bool)
    signatures = tech[has_tech]
    return {
        "profile_rows": int(len(profiles)),
        "profiles_with_techniques": int(has_tech.sum()),
        "unique_technique_signatures": int(signatures.nunique()),
        "profiles_with_documented_cves": int(with_cve.sum()) if len(with_cve) else 0,
        "columns": list(profiles.columns),
    }


def describe_normalized(path: Path = Path("Dataset Normalized.csv")) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    reg = describe_registry(df)
    reg["source_file"] = str(path)
    return reg


def merge_coverage(registry: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    reg = registry.copy()
    reg["gang_key"] = reg["gang"].map(actor_key)
    key_col = "gang_key" if "gang_key" in profiles.columns else "actor_key"
    prof = profiles.set_index(key_col)
    keys = reg["gang_key"]
    has_profile = keys.isin(prof.index)
    tech_series = keys.map(
        lambda k: clean_technique_csv(prof.at[k, "ttp_techniques"]) if k in prof.index else ""
    )
    has_tech = tech_series.str.strip().ne("")
    return {
        "rows_with_profile_row": int(has_profile.sum()),
        "rows_with_non_empty_techniques": int(has_tech.sum()),
        "share_with_techniques": float(has_tech.mean()) if len(reg) else 0.0,
    }


def to_markdown(payload: dict) -> str:
    lines = [
        "# MLEM dataset descriptor",
        "",
        f"- Workbook: `{payload['workbook']['path']}` ({payload['workbook']['schema']})",
        f"- As-of filter: `{payload.get('as_of_date') or 'none (full workbook)'}`",
        "",
        "## Registry",
        "",
        f"- Rows: **{payload['registry']['rows']}**",
        f"- Distinct gangs: **{payload['registry']['distinct_gangs']}**",
        f"- Date range: **{payload['registry']['date_min']}** → **{payload['registry']['date_max']}**",
        f"- Missing dates: **{payload['registry']['missing_date']}**",
        f"- Exact duplicate rows: **{payload['registry']['duplicate_rows_exact']}**",
        f"- Missing country: **{payload['registry']['missing_country']}**",
        f"- Unknown / missing sector: **{payload['registry']['unknown_sector']}** / "
        f"**{payload['registry']['missing_sector']}**",
        "",
        "### Disclosures by year",
        "",
        "```json",
        json.dumps(payload["registry"]["disclosures_by_year"], indent=2),
        "```",
        "",
        "### Top gangs",
        "",
        "```json",
        json.dumps(payload["registry"]["top_gangs_15"], indent=2),
        "```",
        "",
        "## Gang profiles",
        "",
        f"- Profiles: **{payload['profiles']['profile_rows']}**",
        f"- With ATT&CK techniques: **{payload['profiles']['profiles_with_techniques']}**",
        f"- Unique technique signatures: **{payload['profiles']['unique_technique_signatures']}**",
        f"- With documented CVEs: **{payload['profiles']['profiles_with_documented_cves']}**",
        "",
        "## Registry ↔ profile merge",
        "",
        f"- Rows with matching profile: **{payload['merge']['rows_with_profile_row']}**",
        f"- Rows with non-empty techniques: **{payload['merge']['rows_with_non_empty_techniques']}** "
        f"({payload['merge']['share_with_techniques']:.1%})",
        "",
    ]
    if payload.get("normalized_csv"):
        lines.extend(
            [
                "## Normalized CSV (generated)",
                "",
                f"- Rows: **{payload['normalized_csv']['rows']}**",
                f"- Distinct gangs: **{payload['normalized_csv']['distinct_gangs']}**",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = workbook_summary()
    reg = load_registry()
    profiles = load_gang_profiles()
    payload = {
        "workbook": wb,
        "as_of_date": get_as_of_date(),
        "workbook_path": str(resolve_workbook()),
        "registry": describe_registry(reg),
        "profiles": describe_profiles(profiles),
        "merge": merge_coverage(reg, profiles),
        "normalized_csv": describe_normalized(),
    }
    MD_PATH.write_text(to_markdown(payload), encoding="utf-8")
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()