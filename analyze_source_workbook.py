"""
Analyze Dataset Ransomware.xlsx (source workbook for MLEM).
Writes SOURCE_DATASET_ANALYSIS.md in the repository root.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WORKBOOK = Path("Dataset Ransomware.xlsx")
OUTPUT = Path("SOURCE_DATASET_ANALYSIS.md")


def sheet_overview(xl: pd.ExcelFile) -> list[dict]:
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
    ttp_cols = [c for c in df.columns if c == "TTPS" or str(c).startswith("TTPS.")]
    out: dict = {
        "profile_rows": int(len(df)),
        "gang_column": "Gang name" in df.columns,
        "ttp_columns": len(ttp_cols),
    }
    if "Gang name" in df.columns:
        out["gangs_documented"] = int(df["Gang name"].nunique(dropna=True))
        non_empty = 0
        for _, row in df.iterrows():
            if any(pd.notna(row.get(c)) and str(row.get(c)).strip() for c in ttp_cols):
                non_empty += 1
        out["profiles_with_any_ttp"] = non_empty
    return out


def render_markdown(overview: list[dict], dataset: dict, profile: dict) -> str:
    lines = [
        "# Source workbook analysis",
        "",
        f"File: `{WORKBOOK}`",
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
            "## Dataset sheet (incidents)",
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
            "### Top victim sectors (raw labels)",
            "",
            "```json",
            json.dumps(dataset.get("sector_top_10", {}), indent=2),
            "```",
            "",
            "### Top victim countries",
            "",
            "```json",
            json.dumps(dataset.get("country_top_10", {}), indent=2),
            "```",
            "",
            "### Columns with missing values",
            "",
            "```json",
            json.dumps(dataset.get("null_counts", {}), indent=2),
            "```",
            "",
            "## Ransomware Gang Profile sheet",
            "",
            f"- Profile rows: **{profile.get('profile_rows', 'n/a')}**",
            f"- Documented gang names: **{profile.get('gangs_documented', 'n/a')}**",
            f"- TTP column groups: **{profile.get('ttp_columns', 0)}**",
            f"- Profiles with at least one TTP token: **{profile.get('profiles_with_any_ttp', 0)}**",
            "",
            "## Pipeline role",
            "",
            "The **Dataset** sheet feeds victim-level disclosures (gang, date, sector, country). "
            "The **Ransomware Gang Profile** sheet supplies static MITRE ATT&CK technique lists "
            "merged onto every incident with a matching gang name during `dataset_ML_Formatter.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(WORKBOOK)

    xl = pd.ExcelFile(WORKBOOK)
    overview = sheet_overview(xl)

    dataset_df = pd.read_excel(xl, sheet_name="Dataset")
    dataset_stats = analyze_dataset_sheet(dataset_df)

    profile_stats: dict = {}
    if "Ransomware Gang Profile" in xl.sheet_names:
        profile_df = pd.read_excel(xl, sheet_name="Ransomware Gang Profile")
        profile_stats = analyze_profile_sheet(profile_df)

    OUTPUT.write_text(
        render_markdown(overview, dataset_stats, profile_stats),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()