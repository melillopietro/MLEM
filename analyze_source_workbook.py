"""
Analyze the active MLEM source workbook (DATASETv3.xlsx or legacy).
Writes SOURCE_DATASET_ANALYSIS.md in the repository root.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlem_workbook import (
    detect_schema,
    load_gang_profiles,
    load_registry,
    resolve_workbook,
    workbook_summary,
)

OUTPUT = Path("SOURCE_DATASET_ANALYSIS.md")


def sheet_overview(path: Path) -> list[dict]:
    xl = pd.ExcelFile(path)
    rows: list[dict] = []
    for name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=name)
        rows.append(
            {
                "sheet": name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
            }
        )
    return rows


def analyze_dataset_sheet(df: pd.DataFrame) -> dict:
    out: dict = {}
    out["rows"] = int(len(df))
    out["columns"] = list(df.columns)
    if "gang" in df.columns:
        out["gang_unique"] = int(df["gang"].nunique(dropna=True))
        out["gang_top_15"] = (
            df["gang"].value_counts().head(15).astype(int).to_dict()
        )
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        out["date_parse_failures"] = int(dates.isna().sum())
        if dates.notna().any():
            out["date_min"] = str(dates.min().date())
            out["date_max"] = str(dates.max().date())
            out["by_year"] = (
                dates.dt.year.value_counts().sort_index().astype(int).to_dict()
            )
    if "Victim sectors" in df.columns:
        out["sector_unique"] = int(df["Victim sectors"].nunique(dropna=True))
        out["sector_top_10"] = (
            df["Victim sectors"].value_counts().head(10).astype(int).to_dict()
        )
    if "Victim Country" in df.columns:
        out["country_unique"] = int(df["Victim Country"].nunique(dropna=True))
        out["country_top_10"] = (
            df["Victim Country"].value_counts().head(10).astype(int).to_dict()
        )
    out["null_counts"] = {
        k: int(v) for k, v in df.isna().sum().items() if int(v) > 0
    }
    out["duplicate_rows"] = int(df.duplicated().sum())
    return out


def analyze_profile_sheet(df: pd.DataFrame) -> dict:
    out: dict = {
        "profile_rows": int(len(df)),
        "gangs_documented": int(df["gang"].nunique(dropna=True)),
    }
    if "ttp_techniques" in df.columns:
        has = df["ttp_techniques"].fillna("").astype(str).str.strip().ne("")
        out["profiles_with_techniques"] = int(has.sum())
        out["profiles_with_tactics"] = int(
            df.get("ttp_tactics", "")
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
    out["profile_columns"] = list(df.columns)
    for col in [
        "profile_origin",
        "profile_extortion_type",
        "profile_current_status",
        "profile_ecosystem_phase",
    ]:
        if col in df.columns:
            out[f"{col}_top"] = (
                df[col].astype(str).value_counts().head(5).astype(int).to_dict()
            )
    return out


def render_markdown(
    summary: dict,
    overview: list[dict],
    dataset: dict,
    profile: dict,
) -> str:
    lines = [
        "# Source workbook analysis",
        "",
        f"File: `{summary['path']}` (schema **{summary['schema']}**)",
        "",
        "## Sheets",
        "",
        "| Sheet | Rows | Columns |",
        "|---|---:|---:|",
    ]
    for item in overview:
        lines.append(f"| {item['sheet']} | {item['rows']} | {item['columns']} |")

    lines.extend(
        [
            "",
            "## Registry / Dataset sheet",
            "",
            f"- Rows: **{dataset.get('rows', 'n/a')}**",
            f"- Distinct gangs: **{dataset.get('gang_unique', 'n/a')}**",
            f"- Date range: **{dataset.get('date_min', '?')}** – **{dataset.get('date_max', '?')}**",
            f"- Unparseable dates: **{dataset.get('date_parse_failures', 0)}**",
            f"- Exact duplicate rows: **{dataset.get('duplicate_rows', 0)}**",
            "",
            "### Disclosures by year",
            "",
            "```json",
            json.dumps(dataset.get("by_year", {}), indent=2),
            "```",
            "",
            "### Top gangs",
            "",
            "```json",
            json.dumps(dataset.get("gang_top_15", {}), indent=2),
            "```",
            "",
            "### Columns with missing values",
            "",
            "```json",
            json.dumps(dataset.get("null_counts", {}), indent=2),
            "```",
            "",
            "## Gang profile sheet",
            "",
            f"- Profile rows: **{profile.get('profile_rows', 'n/a')}**",
            f"- Documented gang names: **{profile.get('gangs_documented', 'n/a')}**",
            f"- Profiles with ATT&CK techniques: **{profile.get('profiles_with_techniques', 'n/a')}**",
            f"- Profiles with ATT&CK tactics: **{profile.get('profiles_with_tactics', 'n/a')}**",
            "",
            "### Profile column groups",
            "",
            "```json",
            json.dumps(profile.get("profile_columns", []), indent=2),
            "```",
            "",
            "## Pipeline role",
            "",
            "Disclosure rows feed `Dataset Normalized.csv`. Gang profiles supply "
            "`TTP_Techniques` / `TTP_Tactics` (v3) or legacy wide TTPS columns, plus "
            "metadata (origin, extortion type, CVE list, lineage, status) merged in "
            "`dataset_ML_Formatter.py` using a case-insensitive `actor_key`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = resolve_workbook()
    summary = workbook_summary()
    overview = sheet_overview(path)
    dataset_stats = analyze_dataset_sheet(load_registry(path))
    profile_stats = analyze_profile_sheet(load_gang_profiles(path))
    OUTPUT.write_text(
        render_markdown(summary, overview, dataset_stats, profile_stats),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()