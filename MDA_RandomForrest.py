
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import os

# === CONFIG ===
model_path = "classificatore_output/best_model_min20.pkl"
dataset_path = "final_ml_dataset_encoded.csv"
features_path = "features_min20.pkl"
output_csv = "top_20_features_mda.csv"
output_plot = "top_20_features_mda.png"
target_col = "label_gang"

# === 1. Carica tutto ===
assert os.path.exists(model_path), "Modello non trovato"
assert os.path.exists(dataset_path), "Dataset non trovato"
assert os.path.exists(features_path), "Feature list non trovata"

print("📦 Carico modello e dataset...")
model = joblib.load(model_path)
df = pd.read_csv(dataset_path)
feature_names = joblib.load(features_path)

# === 2. Estrai X e y ===
X = df[feature_names]
y = df[target_col]

print(f"✅ Dataset allineato: {X.shape[0]} righe, {X.shape[1]} feature")

# === 3. Calcolo MDA ===
print("⚙️ Calcolo MDA...")
result = permutation_importance(
    model, X, y, n_repeats=10, random_state=42, n_jobs=-1
)

# === 4. Costruisci DataFrame dei risultati ===
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "mean_importance": result.importances_mean,
    "std_importance": result.importances_std
}).sort_values(by="mean_importance", ascending=False)

if importance_df["mean_importance"].sum() == 0:
    print("⚠️ Tutte le importanze risultano nulle. Verifica l'allineamento delle feature.")
else:
    top_20 = importance_df.head(20)
    top_20.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(top_20["Feature"], top_20["mean_importance"], color="skyblue")
    plt.xlabel("Mean Decrease in Accuracy")
    plt.title("Top 20 Feature Importance (MDA)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

    print(f"✅ MDA salvata in: {output_csv}")
    print(f"📊 Grafico salvato in: {output_plot}")
