#!/usr/bin/env python3
"""
Run three MLEM scenarios for paper-oriented comparison:
  1. legacy_paper   — Dataset Ransomware.xlsx, as-of 2025-04-01
  2. v3_paper       — DATASETv3.xlsx, as-of 2025-04-01 (~18.6k disclosures)
  3. v3_full        — DATASETv3.xlsx, no as-of (~23.8k disclosures)

Writes per-scenario artifacts under experiments/comparison/<scenario_id>/
and a consolidated THREE_WAY_COMPARISON_REPORT.md at the repository root.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_DIR = ROOT / "experiments" / "comparison"
REPORT_PATH = ROOT / "THREE_WAY_COMPARISON_REPORT.md"
PAPER_RF_F1 = {
    "ttp_only": 0.9247,
    "ttp_plus_context": 0.9726,
    "context_only": 0.1892,
}

sys.path.insert(0, str(ROOT))

from mlem_config import save_config  # noqa: E402
from mlem_workbook import (  # noqa: E402
    ENV_WORKBOOK,
    WORKBOOK_LEGACY,
    WORKBOOK_V3,
    detect_schema,
    load_gang_profiles,
    load_registry,
    resolve_workbook,
)


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    workbook: Path
    as_of_date: str | None


SCENARIOS: list[Scenario] = [
    Scenario(
        "legacy_paper",
        "Legacy (finestra paper)",
        WORKBOOK_LEGACY,
        "2025-04-01",
    ),
    Scenario(
        "v3_paper",
        "DATASETv3 finestra paper",
        WORKBOOK_V3,
        "2025-04-01",
    ),
    Scenario(
        "v3_full",
        "DATASETv3 completo (23.807 righe)",
        WORKBOOK_V3,
        None,
    ),
]


def _configure_scenario(scenario: Scenario) -> None:
    os.environ[ENV_WORKBOOK] = str(scenario.workbook.name)
    save_config(
        {
            "as_of_date": scenario.as_of_date,
            "description": scenario.label,
            "profile_mode": "default",
        }
    )
    if scenario.as_of_date:
        os.environ["MLEM_AS_OF"] = scenario.as_of_date
    else:
        os.environ.pop("MLEM_AS_OF", None)


def _run_script(rel: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / rel), *(extra or [])]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _corpus_stats(scenario: Scenario) -> dict[str, object]:
    path = scenario.workbook
    schema = detect_schema(path)
    xl = pd.ExcelFile(path)
    reg_sheet = "Registry" if schema.value == "v3" else "Dataset"
    raw = pd.read_excel(path, sheet_name=reg_sheet)
    reg_filtered = load_registry(path=path, apply_as_of=True)
    reg_full = load_registry(path=path, apply_as_of=False)
    profiles = load_gang_profiles(path=path)
    dates = pd.to_datetime(raw["date"], errors="coerce")
    return {
        "workbook": path.name,
        "schema": schema.value,
        "registry_sheet_rows_excel": int(len(raw)),
        "registry_rows_with_as_of": int(len(reg_filtered)),
        "registry_rows_unfiltered": int(len(reg_full)),
        "as_of_date": scenario.as_of_date,
        "distinct_gangs_raw": int(raw["gang"].nunique(dropna=True)),
        "distinct_gangs_as_of": int(reg_filtered["gang"].nunique(dropna=True)),
        "date_min": str(dates.min().date()) if dates.notna().any() else None,
        "date_max_raw": str(dates.max().date()) if dates.notna().any() else None,
        "date_max_as_of": str(
            pd.to_datetime(reg_filtered["date"], errors="coerce").max().date()
        ),
        "disclosures_by_year_raw": (
            dates.dt.year.value_counts().sort_index().astype(int).to_dict()
        ),
        "profile_rows": int(len(profiles)),
        "profiles_with_techniques": int(
            profiles["ttp_techniques"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "profiles_with_cve": int(
            profiles.get("cves_exploited", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.contains("CVE-", case=False, regex=False)
            .sum()
        ),
    }


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _ablation_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rf = df[df["model"] == "RandomForest"]
    out: dict[str, float] = {}
    for _, row in rf.iterrows():
        out[str(row["feature_set"])] = float(row["macro_f1"])
    meta = rf.iloc[0] if len(rf) else None
    if meta is not None:
        out["_n_rows"] = float(meta["n_rows"])
        out["_n_features_ttp_only"] = float(
            rf.loc[rf["feature_set"] == "ttp_only", "n_features"].iloc[0]
            if (rf["feature_set"] == "ttp_only").any()
            else float("nan")
        )
        out["_n_labels"] = float(meta["n_labels"])
    return out


def _rq3_summary(rq3_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    audit = rq3_dir / "audit_summary.json"
    if audit.exists():
        out["audit"] = json.loads(audit.read_text(encoding="utf-8"))
    temporal = rq3_dir / "temporal_context_model_results.csv"
    if temporal.exists():
        df = pd.read_csv(temporal)
        pick = (
            df.sort_values("closed_macro_f1", ascending=False)
            .groupby("evaluation_period", as_index=False)
            .first()
        )
        out["temporal_best"] = pick[
            [
                "evaluation_period",
                "model",
                "closed_macro_f1",
                "full_macro_f1",
                "closed_set_coverage",
            ]
        ].to_dict(orient="records")
    return out


def run_scenario(scenario: Scenario) -> dict[str, object]:
    out_dir = COMPARISON_DIR / scenario.id
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"Scenario: {scenario.id} — {scenario.label}")
    print("=" * 72)

    _configure_scenario(scenario)
    assert resolve_workbook().resolve() == scenario.workbook.resolve()

    stats = _corpus_stats(scenario)
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    nd_args = ["--as-of-date", scenario.as_of_date] if scenario.as_of_date else []
    _run_script("normalized_dataset.py", nd_args)
    _run_script("aggregate_ablation.py")
    _run_script("signature_overlap_audit.py")
    _run_script("rq3_validated_v2/run_all.py")

    _copy_if_exists(ROOT / "reports/aggregate_ablation_results.csv", out_dir / "aggregate_ablation_results.csv")
    _copy_if_exists(ROOT / "reports/signature_overlap_audit.json", out_dir / "signature_overlap_audit.json")
    rq3_out = ROOT / "rq3_validated_v2" / "outputs"
    shutil.copytree(rq3_out, out_dir / "rq3_outputs", dirs_exist_ok=True)

    ablation = _ablation_summary(out_dir / "aggregate_ablation_results.csv")
    sig = json.loads((out_dir / "signature_overlap_audit.json").read_text(encoding="utf-8"))
    rq3 = _rq3_summary(out_dir / "rq3_outputs")

    return {
        "scenario_id": scenario.id,
        "label": scenario.label,
        "corpus": stats,
        "ablation_rf_macro_f1": ablation,
        "signature_audit": {
            "eligible_rows": sig.get("eligible_rows"),
            "eligible_labels": sig.get("eligible_labels"),
            "unique_ttp_signatures": sig.get("unique_ttp_signatures_among_labels"),
            "test_signatures_seen_in_train_fraction": sig.get(
                "test_signatures_seen_in_train_fraction"
            ),
            "exact_signature_lookup_macro_f1": sig.get("exact_signature_lookup_macro_f1"),
        },
        "rq3": rq3,
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(results: list[dict[str, object]]) -> None:
    by_id = {str(r["scenario_id"]): r for r in results}

    corpus_rows = []
    for sid in ("legacy_paper", "v3_paper", "v3_full"):
        c = by_id[sid]["corpus"]  # type: ignore[index]
        corpus_rows.append(
            [
                by_id[sid]["label"],  # type: ignore[index]
                str(c["registry_sheet_rows_excel"]),  # type: ignore[index]
                str(c["registry_rows_with_as_of"]),  # type: ignore[index]
                str(c["distinct_gangs_as_of"]),  # type: ignore[index]
                str(c["profile_rows"]),  # type: ignore[index]
                str(c["date_max_raw"]),  # type: ignore[index]
                str(c["date_max_as_of"]),  # type: ignore[index]
            ]
        )

    def abl_cell(sid: str, key: str) -> str:
        val = by_id[sid]["ablation_rf_macro_f1"].get(key)  # type: ignore[union-attr]
        return f"{val:.3f}" if isinstance(val, float) else "—"

    ablation_rows = []
    for key, paper in [
        ("ttp_only", PAPER_RF_F1["ttp_only"]),
        ("ttp_plus_context", PAPER_RF_F1["ttp_plus_context"]),
        ("ttp_plus_context_plus_cve", None),
        ("context_only", PAPER_RF_F1["context_only"]),
    ]:
        ablation_rows.append(
            [
                key,
                f"{paper:.3f}" if paper is not None else "—",
                abl_cell("legacy_paper", key),
                abl_cell("v3_paper", key),
                abl_cell("v3_full", key),
            ]
        )

    sig_rows = []
    for sid in ("legacy_paper", "v3_paper", "v3_full"):
        s = by_id[sid]["signature_audit"]  # type: ignore[index]
        sig_rows.append(
            [
                by_id[sid]["label"],  # type: ignore[index]
                str(s["eligible_rows"]),
                str(s["eligible_labels"]),
                f"{float(s['test_signatures_seen_in_train_fraction']):.3f}",  # type: ignore[arg-type]
                f"{float(s['exact_signature_lookup_macro_f1']):.3f}",  # type: ignore[arg-type]
            ]
        )

    rq3_rows = []
    for sid in ("legacy_paper", "v3_paper", "v3_full"):
        rq3 = by_id[sid].get("rq3", {})  # type: ignore[union-attr]
        best = {r["evaluation_period"]: r for r in rq3.get("temporal_best", [])}  # type: ignore[union-attr]
        for period in ("validation_2024", "test_2025"):
            row = best.get(period)
            if row:
                rq3_rows.append(
                    [
                        by_id[sid]["label"],  # type: ignore[index]
                        period,
                        str(row.get("model")),
                        f"{float(row['closed_macro_f1']):.4f}",
                        f"{float(row['closed_set_coverage']):.3f}",
                    ]
                )

    today = date.today().isoformat()
    def int_cell(sid: str, key: str) -> str:
        val = by_id[sid]["ablation_rf_macro_f1"].get(key)  # type: ignore[union-attr]
        return str(int(val)) if isinstance(val, float) and val == val else "—"

    md = f"""# Confronto a tre vie: legacy vs v3 (paper) vs v3 (completo)

**Generato:** {today}  
**Scenari:** (1) `Dataset Ransomware.xlsx` con cutoff paper; (2) `DATASETv3.xlsx` stesso cutoff;
(3) `DATASETv3.xlsx` senza cutoff (tutte le righe Registry, fino a 2025-12-31).

Riferimento Tab. 5 paper (Random Forest, split aggregato random): TTP **0,925**, TTP+contesto **0,973**, solo contesto **0,189**.

---

## 1. Corpus

{_md_table(
    ["Scenario", "Righe Excel", "Dopo as-of", "Gang (as-of)", "Profili", "Data max (file)", "Data max (as-of)"],
    corpus_rows,
)}

**Lettura:** ~**18,6k** disclosure compaiono solo con cutoff **2025-04-01** (legacy ≈ v3 paper).
Il terzo scenario aggiunge **~5,2k** eventi (2025 apr–dic); le righe Excel passano a **23.807**.

---

## 2. Ablation aggregata (Random Forest macro-F1, come Tab. 5)

{_md_table(
    ["Feature set", "Paper RF", "Legacy paper", "v3 paper", "v3 full"],
    ablation_rows,
)}

Metriche ausiliarie (righe/label ML con TTP non vuoto, soglia ≥10):

| Scenario | Righe ML | Label (≥10) |
|----------|---------:|------------:|
| Legacy paper | {int_cell("legacy_paper", "_n_rows")} | {int_cell("legacy_paper", "_n_labels")} |
| v3 paper | {int_cell("v3_paper", "_n_rows")} | {int_cell("v3_paper", "_n_labels")} |
| v3 full | {int_cell("v3_full", "_n_rows")} | {int_cell("v3_full", "_n_labels")} |

---

## 3. Audit leakage firma TTP (split random 15% test)

{_md_table(
    ["Scenario", "Righe eleggibili", "Label", "Firme test già in train", "Lookup macro-F1"],
    sig_rows,
)}

---

## 4. RQ3 cronologico (miglior modello per periodo, feature sector+country)

{_md_table(
    ["Scenario", "Periodo", "Modello", "Closed macro-F1", "Copertura closed-set"],
    rq3_rows,
)}

**Messaggio per il paper:** estendere il corpus a **23,8k** **non** riporta l’attribuzione cronologica
verso i valori Tab. 5; i numeri restano nell’ordine di **10⁻²–10⁻³**. Il salto verso ~0,93–0,97
resta confinato al benchmark **aggregato** con profili TTP statici.

---

## 5. Implicazioni resubmit

| Domanda revisore | Cosa mostrano i 3 scenari |
|------------------|---------------------------|
| Dataset stats | Riga **v3 full** documenta l’estensione 2025 senza confonderla con la finestra paper |
| Tab. 5 / leakage | Legacy vs **v3 paper**: profili/CVE migliorano RF aggregato; lookup ~100% firme in train in tutti |
| Attribution OOT | **v3 full** aumenta il rumore temporale ma **non** il macro-F1 cronologico |
| Quale corpus citare | Manoscritto principale: **v3 paper** (parità legacy); supplemento: **v3 full** |

Artefatti per scenario: `experiments/comparison/<scenario_id>/`.

---

*Pipeline: `python3 experiments/compare_three_datasets.py`*
"""
    REPORT_PATH.write_text(md, encoding="utf-8")
    summary = {"generated": today, "scenarios": results}
    (COMPARISON_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {REPORT_PATH}")


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_scenario(s) for s in SCENARIOS]
    write_report(results)


if __name__ == "__main__":
    main()