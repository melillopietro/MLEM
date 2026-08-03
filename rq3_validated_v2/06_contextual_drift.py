from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency

from common import OUTPUT_DIR, load_disclosures

YEARS = list(range(2020, 2026))
MIN_GANG_YEAR_RECORDS = 5


def entropy(values: pd.Series) -> float:
    p = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(p * np.log2(p)).sum()) if len(p) else float("nan")


def concentration(values: pd.Series, k: int) -> float:
    shares = values.value_counts(normalize=True)
    return float(shares.head(k).sum()) if len(shares) else float("nan")


def distribution_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for (year, gang), part in df.groupby(["year", "gang"]):
        counts = part[column].value_counts()
        rows.append(
            {
                "dimension": column,
                "year": year,
                "gang": gang,
                "records": len(part),
                "categories": int(counts.size),
                "dominant_category": counts.index[0],
                "dominant_share": float(counts.iloc[0] / len(part)),
                "top3_share": concentration(part[column], 3),
                "entropy_bits": entropy(part[column]),
                "eligible_min_5_records": len(part) >= MIN_GANG_YEAR_RECORDS,
            }
        )
    return pd.DataFrame(rows)


def adjacent_drift(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for gang, actor in df.groupby("gang"):
        actor_years = sorted(set(actor["year"]).intersection(YEARS))
        for y1, y2 in zip(actor_years, actor_years[1:]):
            if y2 != y1 + 1:
                continue
            a = actor[actor["year"] == y1][column]
            b = actor[actor["year"] == y2][column]
            categories = sorted(set(a).union(b))
            ca = a.value_counts().reindex(categories, fill_value=0)
            cb = b.value_counts().reindex(categories, fill_value=0)
            pa = ca / ca.sum()
            pb = cb / cb.sum()
            js = float(jensenshannon(pa, pb, base=2) ** 2)
            table = np.vstack([ca.to_numpy(), cb.to_numpy()])
            chi2 = p = cramers_v = np.nan
            test_valid = bool(
                len(categories) >= 2
                and len(a) >= MIN_GANG_YEAR_RECORDS
                and len(b) >= MIN_GANG_YEAR_RECORDS
            )
            if test_valid:
                nonzero = table.sum(axis=0) > 0
                table = table[:, nonzero]
                if table.shape[1] >= 2:
                    chi2, p, _, _ = chi2_contingency(table)
                    n = table.sum()
                    cramers_v = float(np.sqrt(chi2 / n)) if n else np.nan
                else:
                    test_valid = False
            rows.append(
                {
                    "dimension": column,
                    "gang": gang,
                    "year_from": y1,
                    "year_to": y2,
                    "records_from": len(a),
                    "records_to": len(b),
                    "categories_union": len(categories),
                    "jensen_shannon_divergence": js,
                    "chi_square": chi2,
                    "p_value": p,
                    "cramers_v": cramers_v,
                    "chi_square_test_valid": test_valid,
                }
            )
    return pd.DataFrame(rows)


def annual_composition(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df.groupby(["year", column]).size().rename("records").reset_index()
    counts["annual_share"] = counts["records"] / counts.groupby("year")["records"].transform("sum")
    counts.insert(0, "dimension", column)
    return counts


def main() -> None:
    df = load_disclosures(deduplicate=True)
    df = df[df["year"].isin(YEARS)].copy()
    dimensions = ["victim_sector", "victim_country"]

    stats = pd.concat([distribution_stats(df, c) for c in dimensions], ignore_index=True)
    drift = pd.concat([adjacent_drift(df, c) for c in dimensions], ignore_index=True)
    composition = pd.concat([annual_composition(df, c) for c in dimensions], ignore_index=True)

    stats.to_csv(OUTPUT_DIR / "contextual_distribution_by_gang_year.csv", index=False)
    drift.to_csv(OUTPUT_DIR / "contextual_adjacent_year_drift.csv", index=False)
    composition.to_csv(OUTPUT_DIR / "contextual_annual_composition.csv", index=False)

    print("Contextual drift analysis completed")
    print(f"- gang-year distribution rows: {len(stats)}")
    print(f"- adjacent-year comparisons: {len(drift)}")
    print(f"- valid chi-square comparisons: {int(drift['chi_square_test_valid'].sum())}")


if __name__ == "__main__":
    main()
