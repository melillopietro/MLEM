"""
Run configuration for reproducible MLEM pipelines (as-of date, profile mode).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "mlem_run_config.json"
ENV_AS_OF = "MLEM_AS_OF"


def default_config() -> dict[str, Any]:
    return {
        "as_of_date": None,
        "profile_mode": "default",
        "description": "",
    }


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_PATH
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        merged = default_config()
        merged.update(data)
        return merged
    return default_config()


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    path = path or CONFIG_PATH
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    return path


def get_as_of_date() -> str | None:
    env = os.environ.get(ENV_AS_OF, "").strip()
    if env:
        return env
    return load_config().get("as_of_date")


def as_of_timestamp():
    raw = get_as_of_date()
    if not raw:
        return None
    return datetime.strptime(str(raw)[:10], "%Y-%m-%d")


def apply_as_of_frame(df, date_column: str = "date"):
    cutoff = as_of_timestamp()
    if cutoff is None:
        return df
    out = df.copy()
    out[date_column] = pd_to_datetime_series(out[date_column])
    return out[out[date_column] <= cutoff].copy()


def pd_to_datetime_series(series):
    import pandas as pd

    return pd.to_datetime(series, errors="coerce")


def set_as_of_from_cli(value: str | None) -> None:
    config = load_config()
    config["as_of_date"] = value
    save_config(config)
    if value:
        os.environ[ENV_AS_OF] = value
    elif ENV_AS_OF in os.environ:
        del os.environ[ENV_AS_OF]