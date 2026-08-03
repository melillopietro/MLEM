from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
(SCRIPT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
SCRIPTS = [
    "01_audit_dataset.py",
    "02_build_actor_profiles.py",
    "03_actor_similarity.py",
    "04_temporal_context_classification.py",
    "06_contextual_drift.py",
    "05_generate_rq3_report.py",
]
OPTIONAL_SCRIPTS = {"03_actor_similarity.py"}


def main() -> None:
    for script in SCRIPTS:
        path = SCRIPT_DIR / script
        print("\n" + "=" * 80, flush=True)
        print(f"Running {script}", flush=True)
        print("=" * 80, flush=True)
        try:
            subprocess.run([sys.executable, str(path)], check=True, cwd=SCRIPT_DIR)
        except subprocess.CalledProcessError:
            if script not in OPTIONAL_SCRIPTS:
                raise
            print(
                f"Optional step {script} was skipped because an optional dependency "
                "is unavailable. Temporal experiments continue.",
                flush=True,
            )
    print("\nValidated RQ3 workflow completed successfully.")
    print(f"Outputs: {SCRIPT_DIR / 'outputs'}")


if __name__ == "__main__":
    main()
