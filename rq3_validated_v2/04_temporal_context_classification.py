from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from common import (
    MIN_TRAIN_SAMPLES,
    OUTPUT_DIR,
    RANDOM_STATE,
    bootstrap_metric_ci,
    load_disclosures,
    write_json,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

# Primary out-of-time features must be observable for the individual disclosure.
# Calendar attributes are retained only as a sensitivity analysis.
FEATURE_GROUPS = {
    "sector_only": ["victim_sector"],
    "country_only": ["victim_country"],
    "sector_country": ["victim_sector", "victim_country"],
    "sector_country_calendar_sensitivity": [
        "victim_sector",
        "victim_country",
        "month",
        "quarter",
        "dayofweek",
    ],
}
PRIMARY_FEATURE_SET = "sector_country"


def build_model(features: list[str], model_name: str = "RandomForest") -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                features,
            )
        ],
        remainder="drop",
    )
    if model_name == "MajorityBaseline":
        model = DummyClassifier(strategy="most_frequent")
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            C=1.0,
            max_iter=400,
            solver="saga",
            tol=1e-3,
            class_weight=None,
            random_state=RANDOM_STATE,
        )
    elif model_name == "LogisticRegressionBalanced":
        model = LogisticRegression(
            C=1.0,
            max_iter=400,
            solver="saga",
            tol=1e-3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "RandomForestUnweighted":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return Pipeline([("prep", prep), ("model", model)])


def metrics(y_true, y_pred, bootstrap: bool = True) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if bootstrap:
        lo, hi = bootstrap_metric_ci(
            y_true,
            y_pred,
            lambda a, b: f1_score(a, b, average="macro", zero_division=0),
            n_bootstrap=500,
        )
        result["macro_f1_ci95_low"] = lo
        result["macro_f1_ci95_high"] = hi
    return result


def temporal_partition(
    df: pd.DataFrame, train_years: list[int], test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_all = df[df["year"].isin(train_years)].copy()
    test_all = df[df["year"] == test_year].copy()
    counts = train_all["gang"].value_counts()
    known = sorted(counts[counts >= MIN_TRAIN_SAMPLES].index)
    train = train_all[train_all["gang"].isin(known)].copy()
    closed = test_all[test_all["gang"].isin(known)].copy()
    unseen = test_all[~test_all["gang"].isin(known)].copy()
    return train, closed, unseen, known


def fit_and_evaluate(
    train: pd.DataFrame,
    closed: pd.DataFrame,
    unseen: pd.DataFrame,
    features: list[str],
    model_name: str,
    bootstrap: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    model = build_model(features, model_name)
    model.fit(train[features], train["gang"])

    pred_closed = model.predict(closed[features]) if len(closed) else np.array([])
    closed_metrics = metrics(closed["gang"], pred_closed, bootstrap) if len(closed) else {}

    full = pd.concat([closed, unseen], ignore_index=True)
    pred_full = model.predict(full[features]) if len(full) else np.array([])
    # Unseen labels can never be emitted by this closed-set classifier. Computing the
    # full-test metrics on the original strings therefore penalises every unseen case.
    full_metrics = metrics(full["gang"], pred_full, bootstrap) if len(full) else {}

    predictions = full[["date", "year", "gang", "victim_sector", "victim_country"]].copy()
    predictions["prediction"] = pred_full
    predictions["label_seen_in_training"] = predictions["gang"].isin(
        set(train["gang"])
    )
    predictions["correct"] = predictions["gang"] == predictions["prediction"]

    row: dict[str, object] = {
        "model": model_name,
        "feature_set": next(
            (name for name, cols in FEATURE_GROUPS.items() if cols == features),
            ",".join(features),
        ),
        "train_records": len(train),
        "closed_test_records": len(closed),
        "unseen_test_records": len(unseen),
        "full_test_records": len(full),
        "train_labels": int(train["gang"].nunique()),
        "closed_test_labels": int(closed["gang"].nunique()),
        "unseen_test_labels": int(unseen["gang"].nunique()),
        "closed_set_coverage": float(len(closed) / len(full)) if len(full) else np.nan,
    }
    row.update({f"closed_{k}": v for k, v in closed_metrics.items()})
    row.update({f"full_{k}": v for k, v in full_metrics.items()})
    return row, predictions


def main_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation, validation_unseen, known = temporal_partition(
        df, TRAIN_YEARS, VALIDATION_YEAR
    )
    # The same training eligibility policy is carried unchanged to the isolated 2025 test.
    test_all = df[df["year"] == TEST_YEAR].copy()
    test = test_all[test_all["gang"].isin(known)].copy()
    test_unseen = test_all[~test_all["gang"].isin(known)].copy()

    pd.Series(validation_unseen["gang"].value_counts(), name="records").to_csv(
        OUTPUT_DIR / "validation_unseen_labels.csv"
    )
    pd.Series(test_unseen["gang"].value_counts(), name="records").to_csv(
        OUTPUT_DIR / "test_unseen_labels.csv"
    )

    summary = {
        "train_years": TRAIN_YEARS,
        "validation_year": VALIDATION_YEAR,
        "test_year": TEST_YEAR,
        "test_period_is_partial": True,
        "test_date_max": str(test_all["date"].max().date()),
        "minimum_train_samples_per_label": MIN_TRAIN_SAMPLES,
        "train_records": len(train),
        "validation_known_records": len(validation),
        "validation_unseen_records": len(validation_unseen),
        "test_known_records": len(test),
        "test_unseen_records": len(test_unseen),
        "known_train_labels": len(known),
        "validation_known_labels": int(validation["gang"].nunique()),
        "validation_unseen_labels": int(validation_unseen["gang"].nunique()),
        "test_known_labels": int(test["gang"].nunique()),
        "test_unseen_labels": int(test_unseen["gang"].nunique()),
        "known_labels": known,
    }
    write_json(OUTPUT_DIR / "temporal_split_summary.json", summary)

    rows = []
    predictions = []
    candidates = [
        "MajorityBaseline",
        "LogisticRegression",
        "LogisticRegressionBalanced",
        "RandomForestUnweighted",
        "RandomForest",
    ]
    validation_rows = []
    for model_name in candidates:
        row, _ = fit_and_evaluate(
            train,
            validation,
            validation_unseen,
            FEATURE_GROUPS[PRIMARY_FEATURE_SET],
            model_name,
        )
        row["evaluation_period"] = "validation_2024"
        validation_rows.append(row)
    selection = pd.DataFrame(validation_rows)
    selected = (
        selection[selection["model"] != "MajorityBaseline"]
        .sort_values(
            ["closed_macro_f1", "closed_weighted_f1", "closed_accuracy"],
            ascending=False,
        )
        .iloc[0]["model"]
    )
    selection["selected_from_validation"] = selection["model"] == selected
    rows.extend(selection.to_dict("records"))

    # The isolated test is evaluated only for the predeclared baseline and the
    # model selected using 2024 validation metrics.
    for model_name in ["MajorityBaseline", selected]:
        row, pred = fit_and_evaluate(
            train,
            test,
            test_unseen,
            FEATURE_GROUPS[PRIMARY_FEATURE_SET],
            model_name,
        )
        row["evaluation_period"] = "test_2025"
        row["selected_from_validation"] = model_name == selected
        rows.append(row)
        pred["model"] = model_name
        predictions.append(pred)
    pd.DataFrame(
        [{"selection_metric": "closed_macro_f1", "selected_model": selected}]
    ).to_csv(OUTPUT_DIR / "temporal_model_selection.csv", index=False)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def selected_model_name() -> str:
    return pd.read_csv(OUTPUT_DIR / "temporal_model_selection.csv").iloc[0][
        "selected_model"
    ]


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    train, closed, unseen, _ = temporal_partition(df, TRAIN_YEARS, TEST_YEAR)
    rows = []
    for name, features in FEATURE_GROUPS.items():
        row, _ = fit_and_evaluate(
            train, closed, unseen, features, selected_model_name(), bootstrap=False
        )
        row["feature_set"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_origin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_year in [2021, 2022, 2023, 2024, 2025]:
        train_years = list(range(2020, test_year))
        train, closed, unseen, _ = temporal_partition(df, train_years, test_year)
        if len(train) == 0 or len(closed) == 0 or train["gang"].nunique() < 2:
            continue
        row, _ = fit_and_evaluate(
            train,
            closed,
            unseen,
            FEATURE_GROUPS[PRIMARY_FEATURE_SET],
            selected_model_name(),
            bootstrap=False,
        )
        row.update(
            {
                "test_year": test_year,
                "train_year_min": min(train_years),
                "train_year_max": max(train_years),
                "test_period_is_partial": test_year == 2025,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_updated(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, train_years in {
        "frozen_2020_2023": [2020, 2021, 2022, 2023],
        "updated_2020_2024": [2020, 2021, 2022, 2023, 2024],
    }.items():
        train, closed, unseen, _ = temporal_partition(df, train_years, TEST_YEAR)
        row, _ = fit_and_evaluate(
            train,
            closed,
            unseen,
            FEATURE_GROUPS[PRIMARY_FEATURE_SET],
            selected_model_name(),
        )
        row["training_policy"] = name
        row["train_years"] = ",".join(map(str, train_years))
        rows.append(row)
    result = pd.DataFrame(rows)
    frozen = result[result["training_policy"] == "frozen_2020_2023"].iloc[0]
    updated = result[result["training_policy"] == "updated_2020_2024"].iloc[0]
    result["delta_closed_macro_f1_vs_frozen"] = (
        result["closed_macro_f1"] - frozen["closed_macro_f1"]
    )
    result["delta_full_macro_f1_vs_frozen"] = (
        result["full_macro_f1"] - frozen["full_macro_f1"]
    )
    result["delta_closed_set_coverage_vs_frozen"] = (
        result["closed_set_coverage"] - frozen["closed_set_coverage"]
    )
    return result


def plot_rolling(result: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.plot(result["test_year"], result["closed_macro_f1"], marker="o", label="Closed-set")
    plt.plot(result["test_year"], result["full_macro_f1"], marker="s", label="Full temporal test")
    plt.ylabel("Macro-F1")
    plt.xlabel("Test year")
    plt.title("Rolling-origin contextual attribution")
    plt.xticks(result["test_year"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rolling_origin_context_performance.png", dpi=220)
    plt.close()


def main() -> None:
    df = load_disclosures(deduplicate=True)
    models, predictions = main_split(df)
    abl = ablation(df)
    rolling = rolling_origin(df)
    update = frozen_updated(df)

    models.to_csv(OUTPUT_DIR / "temporal_context_model_results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "temporal_test_predictions.csv", index=False)
    abl.to_csv(OUTPUT_DIR / "temporal_context_ablation_results.csv", index=False)
    rolling.to_csv(OUTPUT_DIR / "rolling_origin_context_results.csv", index=False)
    update.to_csv(OUTPUT_DIR / "frozen_vs_updated_results.csv", index=False)
    plot_rolling(rolling)

    print("Temporal context analysis completed")
    print(models.to_string(index=False))
    print("\nRolling-origin:")
    print(rolling.to_string(index=False))
    print("\nFrozen versus updated:")
    print(update.to_string(index=False))


if __name__ == "__main__":
    main()
