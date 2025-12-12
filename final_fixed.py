import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os

# === CONFIGURAZIONE ===
DATA_DIR = "./dataset_split"
MODEL_PATH = "XGBoost_best_model.pkl"
OUTPUT_DIR = "./reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. CARICAMENTO DATI
print("📂 Caricamento dati e modello...")
try:
    # X_train serve solo per recuperare i nomi delle feature
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")) 
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    
    # y_train serve per addestrare il LabelEncoder
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")) 
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    
    feature_names = X_train.columns.tolist()
except FileNotFoundError:
    print("❌ Errore: File dataset non trovati. Assicurati di aver eseguito generate_dataset.py")
    exit()

# 2. CARICAMENTO MODELLO
if not os.path.exists(MODEL_PATH):
    print(f"❌ Errore: Modello {MODEL_PATH} non trovato. Esegui prima il training.")
    exit()

pipeline = joblib.load(MODEL_PATH)

# 3. FIX CRITICO: ALLINEAMENTO ETICHETTE (Encoding)
# Il modello predice numeri (0, 1, 2...), ma y_test contiene stringhe ("LockBit", "Conti"...).
# Dobbiamo convertire y_test in numeri usando lo stesso dizionario del training.
print("🔄 Allineamento etichette (Text -> Number)...")
le = LabelEncoder()
le.fit(y_train["label_gang"])           # Impara la mappa Stringa -> Numero
y_test_enc = le.transform(y_test["label_gang"]) # Converte il test in numeri

# 4. GENERAZIONE REPORT DI CLASSIFICAZIONE
print("📊 Generazione Report Dettagliato per Classe...")
try:
    y_pred = pipeline.predict(X_test)

    # target_names=le.classes_ rimette i nomi reali (LockBit, ecc.) nel report
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)

    # Salvataggio su CSV
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUT_DIR, "full_classification_report.csv"))
    print("✅ Report salvato in reports/full_classification_report.csv")
except Exception as e:
    print(f"⚠️ Errore nella generazione del report: {e}")

# 5. SHAP VALUES (Interpretabilità)
print("🧠 Calcolo SHAP Values (potrebbe richiedere 1-2 minuti)...")

try:
    # Estrazione del modello e dello scaler dalla Pipeline
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps['classifier']
        scaler = pipeline.named_steps['scaler']
    else:
        model = pipeline
        scaler = None

    # Trasformiamo i dati di test come fa la pipeline (Scaling)
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    # Creiamo un DataFrame con i nomi delle feature (per il grafico)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # SHAP TreeExplainer (ottimizzato per XGBoost/RandomForest)
    explainer = shap.TreeExplainer(model)
    
    # Calcoliamo su un sottoinsieme del test set per velocità (es. primi 200 campioni)
    # check_additivity=False evita errori di precisione numerica con XGBoost recenti
    shap_values = explainer.shap_values(X_test_df.iloc[:200], check_additivity=False)

    # Generazione grafico
    plt.figure(figsize=(10, 8))
    
    # Gestione output multiclasse (shap_values potrebbe essere una lista)
    if isinstance(shap_values, list):
        # Se è una lista, prendiamo la classe più frequente (o aggregata) per il summary
        shap.summary_plot(shap_values, X_test_df.iloc[:200], feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test_df.iloc[:200], feature_names=feature_names, show=False)
        
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), bbox_inches='tight', dpi=300)
    print("✅ Grafico SHAP salvato in reports/shap_summary_plot.png")
    
except Exception as e:
    print(f"⚠️ Errore nel calcolo SHAP: {e}")
    print("   (Nota: Se usi XGBoost > 2.0, assicurati che SHAP sia aggiornato all'ultima versione)")

print("\n🎉 Analisi completata!")