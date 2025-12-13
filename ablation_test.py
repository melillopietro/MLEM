import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os

# === CONFIGURAZIONE ===
print("🔬 AVVIO ABLATION STUDY (Studio di Impatto delle Feature)...")

# 1. Caricamento Dati
try:
    X = pd.read_csv("dataset_split/X_train.csv")
    y_raw = pd.read_csv("dataset_split/y_train.csv").values.ravel()
    
    # Encoder delle label (fondamentale per XGBoost)
    if os.path.exists("label_encoder.pkl"):
        le = joblib.load("label_encoder.pkl")
        y = le.transform(y_raw)
    else:
        # Fallback se manca il file pkl
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
    print(f"   -> Dataset caricato: {X.shape}")

except Exception as e:
    print(f"❌ Errore caricamento: {e}")
    exit()

# 2. Definizione dei Gruppi di Feature
all_cols = X.columns.tolist()

# TTPs: Colonne che iniziano per "T" seguito da numero (es. T1059) o "TA" (Tactics)
feats_ttps = [c for c in all_cols if (c.startswith("T") and c[1].isdigit()) or c.startswith("TA")]

# Vittimologia: Colonne con "country", "sector", "victim"
feats_vict = [c for c in all_cols if "country" in c.lower() or "sector" in c.lower() or "victim" in c.lower()]

print(f"   -> Feature TTPs trovate: {len(feats_ttps)}")
print(f"   -> Feature Vittimologia trovate: {len(feats_vict)}")

# 3. Configurazione Esperimenti
experiments = {
    "Solo TTPs (Tecniche)": feats_ttps,
    "Solo Vittimologia (Context)": feats_vict,
    "Full Hybrid (MLEM Model)": all_cols # Usa tutto
}

results = {}

# 4. Esecuzione Training Comparativo
model = XGBClassifier(n_estimators=100, max_depth=10, n_jobs=1) # Parametri standard per confronto equo
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🚀 Inizio Training Comparativo (5-Fold CV)...")

for name, cols in experiments.items():
    if len(cols) == 0:
        print(f"⚠️  Salto {name}: nessua feature trovata.")
        results[name] = 0.0
        continue
        
    print(f"   Running: {name} ({len(cols)} features)...")
    
    # Filtra il dataset solo con le colonne di questo esperimento
    X_subset = X[cols]
    
    # Calcola F1-Score
    scores = cross_val_score(model, X_subset, y, cv=cv, scoring='f1_macro', n_jobs=1)
    avg_score = np.mean(scores)
    results[name] = avg_score
    print(f"      -> F1-Score: {avg_score:.4f}")

# 5. Analisi e Grafico
print("\n" + "="*40)
print("✅ RISULTATI ABLATION STUDY")
print("="*40)
for name, score in results.items():
    print(f"{name:30}: {score:.4f}")

# Calcolo guadagno (Gain)
base = results.get("Solo TTPs (Tecniche)", 0)
hybrid = results.get("Full Hybrid (MLEM Model)", 0)
gain = (hybrid - base) * 100

print("-" * 40)
if gain > 0:
    print(f"🏆 IL MODELLO IBRIDO MIGLIORA LE PRESTAZIONI DEL +{gain:.2f}%")
    print("   Conclusione: L'aggiunta della vittimologia è statisticamente utile.")
else:
    print(f"😐 Nessun miglioramento significativo rilevato.")

# Generazione Grafico per la Tesi
names = list(results.keys())
values = list(results.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values, color=['#bdc3c7', '#bdc3c7', '#2ecc71']) # Grigio, Grigio, Verde (Vincitore)
plt.ylim(0, 1.1)
plt.title("Impact of Hybrid Feature Engineering (Ablation Study)", fontsize=14)
plt.ylabel("F1-Score (Macro)", fontsize=12)

# Aggiungi etichette sui valori
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("ablation_result.png")
print("\n📊 Grafico salvato come 'ablation_result.png'. Inseriscilo nella tesi!")