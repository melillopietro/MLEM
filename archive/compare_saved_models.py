import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# === CONFIG ===
MODEL_DIR = "./5samples"
DATASET_PATH = os.path.join(MODEL_DIR, "final_ml_dataset_encoded.csv")
LABEL_COL = "label_gang"
MIN_SAMPLES = 5
TEST_SIZE = 0.3
RECALL_THRESHOLD = 0.6
OUTPUT_CSV = os.path.join(MODEL_DIR, "model_evaluation_comparison.csv")

# === LOAD DATASET ===
df = pd.read_csv(DATASET_PATH)

# === FILTER BY MIN SAMPLES ===
value_counts = df[LABEL_COL].value_counts()
valid_labels = value_counts[value_counts >= MIN_SAMPLES].index
df = df[df[LABEL_COL].isin(valid_labels)]

# === SPLIT ===
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
index_to_label = dict(enumerate(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=42
)

# === EVALUATE MODELS ===
results = []

for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl") and "best_model" in file:
        model_name = file.replace("_best_model.pkl", "")
        model_path = os.path.join(MODEL_DIR, file)

        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            per_class_recall = [v["recall"] for k, v in report.items() if k.isdigit()]
            min_recall = min(per_class_recall)
            classes_below_threshold = sum(r < RECALL_THRESHOLD for r in per_class_recall)

            result = {
                "model": model_name,
                "accuracy": report["accuracy"],
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "precision_weighted": report["weighted avg"]["precision"],
                "recall_weighted": report["weighted avg"]["recall"],
                "f1_weighted": report["weighted avg"]["f1-score"],
                "lowest_class_recall": min_recall,
                f"n_classes_below_{int(RECALL_THRESHOLD * 100)}%": classes_below_threshold
            }

            # === ADD PER-CLASS METRICS ===
            for i, label in index_to_label.items():
                str_i = str(i)
                if str_i in report:
                    result[f"{label}_precision"] = report[str_i]["precision"]
                    result[f"{label}_recall"] = report[str_i]["recall"]
                    result[f"{label}_f1"] = report[str_i]["f1-score"]

            results.append(result)

        except Exception as e:
            print(f"❌ Errore con il modello {model_name}: {e}")

# === SAVE RESULTS ===
df_results = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Confronto salvato in '{OUTPUT_CSV}' con metriche per classe incluse.")
