from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NORMALIZED = PROJECT_ROOT / "Dataset Normalized.csv"
DATASET_XLSX = PROJECT_ROOT / "Dataset Ransomware.xlsx"
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 10

INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
ATTACK_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
ATTACK_TACTIC_RE = re.compile(r"^TA\d{4}$")
COMPACT_SUBTECHNIQUE_RE = re.compile(r"^T(\d{4})(\d{3})$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def ensure_inputs() -> None:
    missing = [str(p) for p in (DATASET_NORMALIZED, DATASET_XLSX) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n- " + "\n- ".join(missing)
        )


def clean_text(value: object, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = INVISIBLE_RE.sub("", str(value)).strip()
    return text if text else default


def actor_key(value: object) -> str:
    text = clean_text(value, default="").lower()
    return NON_ALNUM_RE.sub("", text)


def build_actor_registry() -> tuple[dict[str, str], pd.DataFrame]:
    """Build a conservative lexical canonicalisation map.

    Only case, whitespace and punctuation variants are merged. Distinct operational
    labels such as LockBit, LockBit2 and LockBit3 remain separate.
    """
    ensure_inputs()
    disclosures = pd.read_csv(DATASET_NORMALIZED, usecols=["gang"])
    disclosure_labels = disclosures["gang"].dropna().map(lambda x: clean_text(x, default=""))
    disclosure_labels = disclosure_labels[disclosure_labels != ""]
    frequencies = disclosure_labels.value_counts()

    profiles = pd.read_excel(DATASET_XLSX, sheet_name="Ransomware Gang Profile", usecols=["Gang name"])
    profile_labels = [clean_text(v, default="") for v in profiles["Gang name"].dropna()]
    profile_labels = [v for v in profile_labels if v]
    profile_by_key: dict[str, str] = {}
    ambiguous_profile_keys: set[str] = set()
    for label in profile_labels:
        key = actor_key(label)
        if key in profile_by_key and profile_by_key[key] != label:
            ambiguous_profile_keys.add(key)
        else:
            profile_by_key[key] = label
    for key in ambiguous_profile_keys:
        profile_by_key.pop(key, None)

    labels_by_key: dict[str, list[str]] = {}
    for label in sorted(set(disclosure_labels)):
        labels_by_key.setdefault(actor_key(label), []).append(label)

    registry: dict[str, str] = {}
    mapping_rows: list[dict[str, object]] = []
    for key, variants in sorted(labels_by_key.items()):
        if key in profile_by_key:
            canonical = profile_by_key[key]
            source = "profile_label"
        else:
            canonical = sorted(variants, key=lambda x: (-int(frequencies.get(x, 0)), x))[0]
            source = "most_frequent_disclosure_label"
        for original in variants:
            registry[original] = canonical
            mapping_rows.append(
                {
                    "actor_key": key,
                    "original_label": original,
                    "canonical_label": canonical,
                    "original_disclosures": int(frequencies.get(original, 0)),
                    "canonical_source": source,
                    "lexical_variant_merged": bool(original != canonical),
                }
            )

    mapping = pd.DataFrame(mapping_rows)
    return registry, mapping


def load_disclosures(
    deduplicate: bool = True,
    canonicalize_labels: bool = True,
) -> pd.DataFrame:
    ensure_inputs()
    df = pd.read_csv(DATASET_NORMALIZED)
    required = {"victim", "gang", "date", "Victim sectors", "Victim Country"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset Normalized.csv is missing columns: {missing}")

    df = df.copy()
    df["gang_original"] = df["gang"].map(lambda x: clean_text(x, default=""))
    if canonicalize_labels:
        registry, _ = build_actor_registry()
        df["gang"] = df["gang_original"].map(lambda x: registry.get(x, x))
    else:
        df["gang"] = df["gang_original"]

    df["victim"] = df["victim"].map(lambda x: clean_text(x, default=""))
    df["victim_sector"] = df["Victim sectors"].map(clean_text)
    df["victim_country"] = df["Victim Country"].map(clean_text)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["gang"] != "") & df["date"].notna()].copy()

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(int)

    if deduplicate:
        dedup_cols = [
            "victim",
            "gang",
            "date",
            "victim_sector",
            "victim_country",
        ]
        df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()

    return df.reset_index(drop=True)


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


def load_actor_profiles() -> tuple[dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    ensure_inputs()
    profiles = pd.read_excel(DATASET_XLSX, sheet_name="Ransomware Gang Profile")
    if "Gang name" not in profiles.columns:
        raise ValueError("The 'Ransomware Gang Profile' sheet has no 'Gang name' column")

    ttp_cols = [c for c in profiles.columns if c == "TTPS" or str(c).startswith("TTPS.")]
    actor_sets: dict[str, set[str]] = {}
    invalid_rows: list[dict[str, str]] = []
    tactic_rows: list[dict[str, str]] = []

    for _, row in profiles.iterrows():
        actor = clean_text(row.get("Gang name"), default="")
        if not actor:
            continue
        techniques: set[str] = set()
        for col in ttp_cols:
            value = row.get(col)
            if pd.isna(value):
                continue
            for raw_token in str(value).split(","):
                canonical, status = canonicalize_attack_token(raw_token)
                if status == "technique" and canonical:
                    techniques.add(canonical)
                elif status == "tactic" and canonical:
                    tactic_rows.append(
                        {
                            "actor_label": actor,
                            "raw_token": raw_token.strip(),
                            "canonical_token": canonical,
                        }
                    )
                elif status == "invalid" and raw_token.strip():
                    invalid_rows.append(
                        {"actor_label": actor, "raw_token": raw_token.strip()}
                    )
        if techniques:
            actor_sets[actor] = techniques

    invalid_df = pd.DataFrame(invalid_rows, columns=["actor_label", "raw_token"])
    tactics_df = pd.DataFrame(
        tactic_rows,
        columns=["actor_label", "raw_token", "canonical_token"],
    )
    return actor_sets, invalid_df, tactics_df


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 500,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores: list[float] = []
    if n == 0:
        return float("nan"), float("nan")
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return path
