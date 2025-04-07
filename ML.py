import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import joblib
from collections import Counter
import os

# === CONFIG ===
input_file = "final_ml_dataset_encoded.csv"
soglie = [2, 5, 10, 20]
output_dir = "classificatore_output"
os.makedirs(output_dir, exist_ok=True)

# === CARICAMENTO ORIGINALE ===
df_originale = pd.read_csv(input_file)

# === RISULTATI ===
f1_results = []

for min_samples in soglie:
    df = df_originale.copy()

    # FILTRA GANG CON >= min_samples
    value_counts = df["label_gang"].value_counts()
    gangs_to_keep = value_counts[value_counts >= min_samples].index
    df = df[df["label_gang"].isin(gangs_to_keep)]

    if len(gangs_to_keep) < 2:
        print(f"⚠️ Soglia {min_samples}: troppo poche classi, salto...")
        continue

    # FEATURES & TARGET
    X = df.drop(columns=["label_gang"])
    y = df["label_gang"]

    # ENCODING
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # SPLIT
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
        )
    except ValueError as e:
        print(f"⚠️ Errore split per soglia {min_samples}: {e}")
        continue

    # CLASSIFICATORE
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # MACRO F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    f1_results.append({"min_samples": min_samples, "macro_f1": macro_f1})
    print(f"🎯 Soglia {min_samples} → Macro F1-score: {macro_f1:.4f}")

    # === REPORT CSV ===
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{output_dir}/classification_report_min{min_samples}.csv")

    # === SALVA MODELLO MIGLIORE ===
    if macro_f1 == max(r["macro_f1"] for r in f1_results):
        joblib.dump(clf, f"{output_dir}/best_model_min{min_samples}.pkl")
        le_path = f"{output_dir}/label_encoder_min{min_samples}.pkl"
        joblib.dump(le, le_path)

# === SALVA F1 SUMMARY ===
f1_df = pd.DataFrame(f1_results)
f1_df.to_csv(f"{output_dir}/macro_f1_comparison.csv", index=False)

print("\n✅ Tutti i risultati salvati nella cartella:", output_dir)
