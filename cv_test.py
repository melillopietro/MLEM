from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import pandas as pd
import numpy as np
import sys

# Configurazioni
N_SPLITS = 5  # Usiamo 5 invece di 10 per evitare errori sulle classi piccole

print("🔄 Avvio Cross-Validation...")

try:
    # 1. Carica Dati
    X = pd.read_csv("dataset_split/X_train.csv")
    y_df = pd.read_csv("dataset_split/y_train.csv") # Questo contiene stringhe!
    
    # 2. Carica Encoder e Modello
    le = joblib.load("label_encoder.pkl")
    model = joblib.load("XGBoost_best_model.pkl")
    
    # 3. CODIFICA LE LABEL (Il passaggio mancante!)
    # Trasforma 'LockBit' -> 42, 'Conti' -> 15, ecc.
    y = le.transform(y_df["label_gang"])
    
    print(f"   -> Dati caricati: {len(X)} campioni.")
    print(f"   -> Esecuzione 5-Fold CV (potrebbe richiedere qualche secondo)...")

    # 4. Esegui CV
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=1)

    # 5. Risultato
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    print("\n" + "="*40)
    print(f"✅ RISULTATO VALIDAZIONE SCIENTIFICA")
    print("="*40)
    print(f"F1-Score Medio:      {mean_score:.4f}")
    print(f"Deviazione Standard: +/- {std_dev:.4f}")
    print("="*40)
    
    if std_dev < 0.05:
        print("CONCLUSIONE: Il modello è ROBUSTO (bassa varianza).")
    else:
        print("CONCLUSIONE: Il modello è INSTABILE (alta varianza).")

except Exception as e:
    print(f"❌ Errore: {e}")
    # Suggerimento debug
    if "y contains new labels" in str(e):
        print("Suggerimento: Hai rigenerato il dataset? Rifai il training nel Tab 1 prima di questo test.")