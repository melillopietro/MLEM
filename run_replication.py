#!/usr/bin/env python3
"""
Run a named replication bundle (see experiments/configs/*.json).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mlem_config import save_config

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "experiments" / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLEM replication bundle.")
    parser.add_argument(
        "config",
        nargs="?",
        default="paper_jisas2026",
        help="Config name (without .json) or path to a JSON file.",
    )
    parser.add_argument(
        "--skip-rq3",
        action="store_true",
        help="Skip rq3_validated_v2/run_all.py",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.suffix:
        config_path = CONFIG_DIR / f"{args.config}.json"
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    import json

    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    save_config(
        {
            "as_of_date": cfg.get("as_of_date"),
            "description": cfg.get("description", ""),
            "profile_mode": cfg.get("profile_mode", "default"),
        }
    )
    print(f"Using config: {config_path.name}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"As-of date: {cfg.get('as_of_date') or 'none'}")

    for step in cfg.get("steps", []):
        if args.skip_rq3 and step.startswith("rq3_validated_v2"):
            continue
        script = ROOT / step
        if not script.exists():
            raise SystemExit(f"Missing script: {script}")
        print("\n" + "=" * 72)
        print(f"Running {step}")
        print("=" * 72)
        cmd = [sys.executable, str(script)]
        if script.name == "normalized_dataset.py" and cfg.get("as_of_date"):
            cmd.extend(["--as-of-date", str(cfg["as_of_date"])])
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nReplication bundle completed.")


if __name__ == "__main__":
    main()