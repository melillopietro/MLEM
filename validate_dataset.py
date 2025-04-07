import pandas as pd
import sys

# === CONFIGURAZIONE ===
# Inserisci il nome del file da validare (CSV o XLSX)
file_path = "temp/final_ml_dataset_encoded.csv"  # o .csv

# === LETTURA FILE ===
try:
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        print("❌ Formato file non supportato. Usa .csv o .xlsx")
        sys.exit(1)
except Exception as e:
    print(f"❌ Errore durante il caricamento del file: {e}")
    sys.exit(1)

# === ANALISI BASE ===
print("\n📊 INFO GENERALI")
print(f"Righe, colonne: {df.shape}")
print(f"Colonne con valori mancanti:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Righe duplicate: {df.duplicated().sum()}")

# === VERIFICA LABEL ===
label_col = "label_gang"
if label_col not in df.columns:
    print(f"❌ Colonna label '{label_col}' non trovata.")
    sys.exit(1)

print(f"\n🏷️ LABEL: '{label_col}'")
print(f"Valori unici: {df[label_col].nunique()}")
print(f"Distribuzione:\n{df[label_col].value_counts().head(10)}")

# === VERIFICA TIPO FEATURE ===
non_numeric = df.drop(columns=[label_col]).select_dtypes(exclude=["int64", "float64", "uint8"]).columns.tolist()
if non_numeric:
    print("\n⚠️ Colonne non numeriche tra le feature (da convertire in numeri):")
    for col in non_numeric:
        print(f" - {col}")
else:
    print("\n✅ Tutte le feature sono numeriche.")

# === SUGGERIMENTO PER LABEL ENCODING ===
print("\n💡 Suggerimento: per usare le label in classificazione ML, puoi applicare LabelEncoder come segue:\n")

print("""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label_gang"])
""")

print("\n✅ Validazione completata.")
