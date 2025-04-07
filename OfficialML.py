import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# === CONFIG ===
folder = "dataset_split"

# === 1. CARICA GLI SPLIT ===
X_train = pd.read_csv(f"{folder}/X_train.csv")
X_test = pd.read_csv(f"{folder}/X_test.csv")
y_train = pd.read_csv(f"{folder}/y_train.csv")["label"]
y_test = pd.read_csv(f"{folder}/y_test.csv")["label"]
label_map = pd.read_csv(f"{folder}/label_mapping.csv")

# === 2. MODELLO ===
clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# === 3. PREDIZIONE ===
y_pred = clf.predict(X_test)

# === 4. VALUTAZIONE ===
print("📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=label_map["gang"]))

macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n🎯 Macro F1-score: {macro_f1:.4f}")

# === 5. CONFUSION MATRIX – TOP 10 ===
top_10_ids = [cls for cls, _ in Counter(y_test).most_common(10)]
top_10_names = label_map.set_index("encoded_label").loc[top_10_ids]["gang"].values

cm = confusion_matrix(y_test, y_pred, labels=top_10_ids)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=top_10_names, yticklabels=top_10_names)
plt.title("Confusion Matrix – Top 10 Gang")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
