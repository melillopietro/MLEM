"""
Load MLEM source workbooks (DATASETv3.xlsx or legacy Dataset Ransomware.xlsx).
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

WORKBOOK_V3 = PROJECT_ROOT / "DATASETv3.xlsx"
WORKBOOK_LEGACY = PROJECT_ROOT / "Dataset Ransomware.xlsx"

SHEET_REGISTRY_V3 = "Registry"
SHEET_PROFILE_V3 = "Gang Profile"
SHEET_DATASET_LEGACY = "Dataset"
SHEET_PROFILE_LEGACY = "Ransomware Gang Profile"

INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
ATTACK_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
ATTACK_TACTIC_RE = re.compile(r"^TA\d{4}$")
COMPACT_SUBTECHNIQUE_RE = re.compile(r"^T(\d{4})(\d{3})$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)

# Gang-profile fields encoded in the aggregate ML pipeline (one-hot).
PROFILE_CATEGORICAL_COLUMNS = [
    "profile_primary_motivation",
    "profile_origin",
    "profile_extortion_type",
    "profile_programming_language",
    "profile_currency",
    "profile_current_status",
    "profile_lineage_type",
    "profile_confidence_level",
    "profile_law_enforcement_impact",
    "profile_ecosystem_phase",
    "profile_dataset_event_correlation",
    "profile_registry_flag",
]


class WorkbookSchema(str, Enum):
    V3 = "v3"
    LEGACY = "legacy"


def resolve_workbook() -> Path:
    if WORKBOOK_V3.exists():
        return WORKBOOK_V3
    if WORKBOOK_LEGACY.exists():
        return WORKBOOK_LEGACY
    raise FileNotFoundError(
        "No source workbook found. Expected DATASETv3.xlsx or Dataset Ransomware.xlsx "
        f"in {PROJECT_ROOT}"
    )


def detect_schema(path: Path | None = None) -> WorkbookSchema:
    path = path or resolve_workbook()
    xl = pd.ExcelFile(path)
    if SHEET_REGISTRY_V3 in xl.sheet_names and SHEET_PROFILE_V3 in xl.sheet_names:
        return WorkbookSchema.V3
    return WorkbookSchema.LEGACY


def clean_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = INVISIBLE_RE.sub("", str(value)).strip()
    return text if text else default


def actor_key(value: object) -> str:
    return NON_ALNUM_RE.sub("", clean_text(value, default="").lower())


def _normalize_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map v3 / legacy profile sheets to a common internal schema."""
    out = df.copy()
    if "Gang name" in out.columns and "gang" not in out.columns:
        out = out.rename(columns={"Gang name": "gang"})
    if "gang" not in out.columns:
        raise ValueError("Profile sheet must contain a gang name column")

    # CVE column has a trailing space in v3 exports.
    cve_col = next(
        (c for c in out.columns if str(c).strip().lower() == "cves exploited"),
        None,
    )
    if cve_col and cve_col != "cves_exploited":
        out = out.rename(columns={cve_col: "cves_exploited"})

    rename_map = {
        "TTP_Techniques": "ttp_techniques",
        "TTP_Tactics": "ttp_tactics",
        "Primary Motivation": "profile_primary_motivation",
        "Origin": "profile_origin",
        "Extortion Type": "profile_extortion_type",
        "Programming Language": "profile_programming_language",
        "Currency": "profile_currency",
        "Current_Status": "profile_current_status",
        "Lineage_Parent": "profile_lineage_parent",
        "Lineage_Type": "profile_lineage_type",
        "Confidence_Level": "profile_confidence_level",
        "Law_Enforcement_Impact": "profile_law_enforcement_impact",
        "Ecosystem_Phase": "profile_ecosystem_phase",
        "Dataset_Event_Correlation": "profile_dataset_event_correlation",
        "Registry": "profile_registry_flag",
        "Aliases": "profile_aliases",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "ttp_techniques" not in out.columns:
        ttp_cols = [c for c in out.columns if c == "TTPS" or str(c).startswith("TTPS.")]
        if ttp_cols:

            def _join_row(row: pd.Series) -> str:
                return ",".join(row.dropna().astype(str))

            out["ttp_techniques"] = out[ttp_cols].apply(_join_row, axis=1)
        else:
            out["ttp_techniques"] = ""
    if "ttp_tactics" not in out.columns:
        out["ttp_tactics"] = ""

    out["gang"] = out["gang"].map(lambda x: clean_text(x, default=""))
    out["actor_key"] = out["gang"].map(actor_key)
    out = out[out["gang"] != ""].drop_duplicates(subset=["actor_key"], keep="first")
    return out


def load_registry(path: Path | None = None, apply_as_of: bool = True) -> pd.DataFrame:
    path = path or resolve_workbook()
    schema = detect_schema(path)
    sheet = SHEET_REGISTRY_V3 if schema == WorkbookSchema.V3 else SHEET_DATASET_LEGACY
    df = pd.read_excel(path, sheet_name=sheet)
    if "Victim sectors" not in df.columns and "Victim Sector" in df.columns:
        df = df.rename(columns={"Victim Sector": "Victim sectors"})
    if apply_as_of:
        from mlem_config import apply_as_of_frame

        df = apply_as_of_frame(df, date_column="date")
    return df


def load_gang_profiles(path: Path | None = None) -> pd.DataFrame:
    path = path or resolve_workbook()
    schema = detect_schema(path)
    sheet = SHEET_PROFILE_V3 if schema == WorkbookSchema.V3 else SHEET_PROFILE_LEGACY
    df = pd.read_excel(path, sheet_name=sheet)
    return _normalize_profile_columns(df)


def canonicalize_attack_token(raw: object) -> tuple[str | None, str]:
    if pd.isna(raw):
        return None, "empty"
    token = INVISIBLE_RE.sub("", str(raw)).upper().strip()
    token = re.sub(r"\s+", "", token)
    token = token.rstrip(";,.")
    if not token:
        return None, "empty"
    compact = COMPACT_SUBTECHNIQUE_RE.fullmatch(token)
    if compact:
        token = f"T{compact.group(1)}.{compact.group(2)}"
    if ATTACK_TECHNIQUE_RE.fullmatch(token):
        return token, "technique"
    if ATTACK_TACTIC_RE.fullmatch(token):
        return token, "tactic"
    return None, "invalid"


def split_attack_tokens(raw: object) -> list[str]:
    if pd.isna(raw):
        return []
    parts = re.split(r"[,;|/\n]+", str(raw))
    return [p.strip() for p in parts if p.strip()]


def tokens_to_csv(tokens: Iterable[str]) -> str:
    return ",".join(sorted(set(tokens)))


def clean_technique_csv(raw: object) -> str:
    cleaned: list[str] = []
    for part in split_attack_tokens(raw):
        token, status = canonicalize_attack_token(part)
        if status == "technique" and token:
            cleaned.append(token)
    return tokens_to_csv(cleaned)


def clean_tactic_csv(raw: object) -> str:
    cleaned: list[str] = []
    for part in split_attack_tokens(raw):
        token, status = canonicalize_attack_token(part)
        if status == "tactic" and token:
            cleaned.append(token)
    return tokens_to_csv(cleaned)


def parse_cve_list(raw: object) -> list[str]:
    if pd.isna(raw):
        return []
    text = str(raw)
    if "unknown" in text.lower() and not CVE_RE.search(text):
        return []
    return sorted({m.group(0).upper() for m in CVE_RE.finditer(text)})


def profile_row_technique_tactics(profile: pd.Series) -> tuple[str, str]:
    tech = clean_technique_csv(profile.get("ttp_techniques", ""))
    tactic = clean_tactic_csv(profile.get("ttp_tactics", ""))
    return tech, tactic


def workbook_summary() -> dict[str, object]:
    path = resolve_workbook()
    schema = detect_schema(path)
    xl = pd.ExcelFile(path)
    reg = load_registry(path)
    prof = load_gang_profiles(path)
    return {
        "path": str(path.name),
        "schema": schema.value,
        "sheets": xl.sheet_names,
        "registry_rows": len(reg),
        "profile_rows": len(prof),
        "registry_gangs": int(reg["gang"].nunique()) if "gang" in reg.columns else None,
        "profiles_with_techniques": int(
            prof["ttp_techniques"].fillna("").astype(str).str.strip().ne("").sum()
        ),
    }