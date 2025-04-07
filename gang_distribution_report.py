import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. CONFIGURAZIONI ===
input_file = "final_ml_dataset_encoded.csv"  # Assicurati che sia nella stessa cartella dello script
output_plot = "gang_distribution.png"
output_csv = "gang_distribution_report.csv"

# === 2. CARICA IL DATASET ===
df = pd.read_csv(input_file)
label_column = "label_gang"

# === 3. CALCOLA DISTRIBUZIONE DELLE GANG ===
gang_distribution = df[label_column].value_counts().reset_index()
gang_distribution.columns = ['Gang', 'Count']
gang_distribution['Percentage'] = (gang_distribution['Count'] / gang_distribution['Count'].sum()) * 100
gang_distribution = gang_distribution.sort_values(by='Count', ascending=False)

# === 4. SALVA DISTRIBUZIONE IN CSV ===
gang_distribution.to_csv(output_csv, index=False)
print(f"✅ Report CSV salvato: {output_csv}")

# === 5. CREA E SALVA IL GRAFICO ===
plt.figure(figsize=(14, 8))
sns.barplot(data=gang_distribution, x='Count', y='Gang', palette='viridis')
plt.title('Distribuzione delle gang ransomware nel dataset')
plt.xlabel('Numero di campioni')
plt.ylabel('Gang')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_plot)
plt.close()
print(f"✅ Grafico salvato come immagine: {output_plot}")
