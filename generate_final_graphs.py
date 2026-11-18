import json
import os
from itertools import cycle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

DATA_DIR = "./dataset_split"
OUTPUT_DIR = "./reports"
ENCODER_FILE = "label_encoder.pkl"
FEATURE_COLS_FROM = os.path.join(DATA_DIR, "X_train.csv")
MODEL_PRIORITY = (
    "best_validation_model.pkl",
    "XGBoost_best_model.pkl",
    "RandomForest_best_model.pkl",
)
MODELS_TO_PLOT = {
    "XGBoost": "XGBoost_best_model.pkl",
    "RandomForest": "RandomForest_best_model.pkl",
}


def resolve_primary_model() -> tuple[str, str] | tuple[None, None]:
    for path in MODEL_PRIORITY:
        if os.path.exists(path):
            return path, path.replace("_best_model.pkl", "").replace("best_validation_model.pkl", "best")
    return None, None


def fitted_estimator(pipeline):
    if hasattr(pipeline, "named_steps"):
        if "clf" in pipeline.named_steps:
            return pipeline.named_steps["clf"]
        if "classifier" in pipeline.named_steps:
            return pipeline.named_steps["classifier"]
    return pipeline


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📂 Caricamento Dati...")
if not os.path.exists(ENCODER_FILE):
    print(f"❌ {ENCODER_FILE} not found. Run training_manager.py first.")
    raise SystemExit(1)

try:
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    cols = pd.read_csv(FEATURE_COLS_FROM, nrows=1).columns.tolist()
    X_test = X_test[cols]
except OSError:
    raise SystemExit(1)

le = joblib.load(ENCODER_FILE)
y_test_enc = le.transform(y_test["label_gang"])
n_classes = len(le.classes_)
y_test_bin = label_binarize(y_test_enc, classes=range(n_classes))

# 1. ROC
print("📊 ROC...")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
colors = cycle(["blue", "green"])
stats = {}
for (name, fname), color in zip(MODELS_TO_PLOT.items(), colors):
    if os.path.exists(fname):
        try:
            pipeline = joblib.load(fname)
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                plt.plot(fpr, tpr, color=color, lw=2, label=name)
                precision, recall, _ = precision_recall_curve(
                    y_test_bin.ravel(), y_score.ravel()
                )
                stats[name] = {"pr": (recall, precision), "color": color}
        except Exception:
            pass
plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.title("ROC")

plt.subplot(1, 2, 2)
for name, data in stats.items():
    plt.plot(
        data["pr"][0], data["pr"][1], color=data["color"], lw=2, label=name
    )
plt.legend()
plt.title("PR Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_ROC_PR_Comparison.png"), dpi=300)

# 2. CM & Features
model_path, model_label = resolve_primary_model()
if model_path:
    meta_path = "best_validation_model.json"
    if model_path == "best_validation_model.pkl" and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            model_label = json.load(handle).get("model", "best")

    print(f"📊 Matrix & Features ({model_label})...")
    pipeline = joblib.load(model_path)
    cm = confusion_matrix(y_test_enc, pipeline.predict(X_test))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, cmap="Blues", vmax=0.8)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Figure_3_Confusion_Matrix_XGBoost.png"),
        dpi=300,
    )

    model = fitted_estimator(pipeline)
    if hasattr(model, "feature_importances_"):
        idx = np.argsort(model.feature_importances_)[::-1][:20]
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(idx)),
            model.feature_importances_[idx],
            align="center",
            color="teal",
        )
        plt.yticks(range(len(idx)), [cols[i] for i in idx])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        plt.savefig(
            os.path.join(OUTPUT_DIR, "Figure_4_Feature_Importance.png"), dpi=300
        )

print("✅ Finito.")