import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# === CONFIG ===
input_file = "final_ml_dataset_encoded.csv"
output_dir = "dataset_split"
label_mapping_path = os.path.join(output_dir, "label_mapping.csv")
min_samples = 10  # ✅ soglia minima per classe

# === CREA CARTELLA OUTPUT SE NON ESISTE ===
os.makedirs(output_dir, exist_ok=True)

# === 1. CARICA IL DATASET ===
df = pd.read_csv(input_file)

# === 2. FILTRA GANG CON ALMENO `min_samples` CAMPIONI ===
value_counts = df["label_gang"].value_counts()
gangs_to_keep = value_counts[value_counts >= min_samples].index
df = df[df["label_gang"].isin(gangs_to_keep)]
print(f"✔️ Classi rimanenti dopo filtro >= {min_samples}: {len(gangs_to_keep)}")

# === 3. SEPARA FEATURES E TARGET ===
X = df.drop(columns=["label_gang"])
y = df["label_gang"]

# === 4. ENCODING LABEL ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 5. TRAIN/TEST SPLIT STRATIFICATO ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# === 6. SALVA I FILE CSV ===
X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["label"]).to_csv(f"{output_dir}/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv(f"{output_dir}/y_test.csv", index=False)

# === 7. CREA E SALVA MAPPING SE NON ESISTE ===
if not os.path.exists(label_mapping_path):
    label_mapping = pd.DataFrame({
        "encoded_label": list(range(len(le.classes_))),
        "gang": le.classes_
    })
    label_mapping.to_csv(label_mapping_path, index=False)
    print(f"✅ 'label_mapping.csv' generato in '{output_dir}/'")
else:
    print(f"✅ 'label_mapping.csv' già presente in '{output_dir}/'")

print("🎯 Dataset pronto per l'addestramento!")
