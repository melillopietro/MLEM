import matplotlib.pyplot as plt
import pandas as pd

# === Inserisci qui i tuoi dati ===
soglie = [2, 5, 10, 20]
macro_f1 = [0.9826, 0.9780, 0.9796, 0.9850]

# === Crea un DataFrame per salvataggio opzionale ===
df_f1 = pd.DataFrame({
    'min_samples': soglie,
    'macro_f1_score': macro_f1
})

# === Salva CSV dei risultati (opzionale) ===
df_f1.to_csv("f1_score_vs_threshold.csv", index=False)

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(soglie, macro_f1, marker='o', linestyle='-', color='royalblue')
plt.title("Macro F1-Score vs Soglia Minima Campioni per Classe", fontsize=14)
plt.xlabel("Soglia (min_samples)", fontsize=12)
plt.ylabel("Macro F1-Score", fontsize=12)
plt.xticks(soglie)
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_score_vs_threshold.png")
plt.show()
