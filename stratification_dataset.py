import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

y_train_path = "y_train.csv"
y_val_path = "y_val.csv"
y_test_path = "y_test.csv"
label_column = "label_gang"
output_csv = "stratificazione_report.csv"
output_plot = "stratificazione_distribuzione.png"

y_train = pd.read_csv(y_train_path)
y_val = pd.read_csv(y_val_path)
y_test = pd.read_csv(y_test_path)

train_dist = y_train[label_column].value_counts(normalize=True)
val_dist = y_val[label_column].value_counts(normalize=True)
test_dist = y_test[label_column].value_counts(normalize=True)

dist_df = pd.DataFrame({
    'Train %': train_dist,
    'Validation %': val_dist,
    'Test %': test_dist
}).fillna(0).sort_values(by='Train %', ascending=False)

dist_df['Val Diff %'] = (dist_df['Validation %'] - dist_df['Train %']) * 100
dist_df['Test Diff %'] = (dist_df['Test %'] - dist_df['Train %']) * 100

dist_df.to_csv(output_csv)
print(f"✅ Report salvato in: {output_csv}")

dataset_sizes = {
    "Train": len(y_train),
    "Validation": len(y_val),
    "Test": len(y_test),
    "Totale": len(y_train) + len(y_val) + len(y_test)
}

print("\n📦 Dimensioni dei set:")
for k, v in dataset_sizes.items():
    print(f"{k}: {v}")

soglia = 2  # percentuale
warning_df = dist_df[(dist_df['Val Diff %'].abs() > soglia) | (dist_df['Test Diff %'].abs() > soglia)]

if not warning_df.empty:
    print("\n⚠️ Classi con differenze > ±2% tra train e val/test:")
    print(warning_df[['Train %', 'Validation %', 'Val Diff %', 'Test %', 'Test Diff %']].round(4))
else:
    print("\n✅ Stratificazione ben mantenuta (nessuna differenza > ±2%)")

top_n = 20
plot_df = dist_df.head(top_n).reset_index().melt(id_vars='index', value_vars=['Train %', 'Validation %', 'Test %'],
                                                 var_name='Split', value_name='Percentuale')
plot_df.rename(columns={'index': 'Gang'}, inplace=True)

plt.figure(figsize=(14, 8))
sns.barplot(data=plot_df, x='Percentuale', y='Gang', hue='Split', palette='Set2')
plt.title(f'Distribuzione % per le prime {top_n} gang nei set')
plt.xlabel("Percentuale")
plt.ylabel("Gang")
plt.legend(title="Split")
plt.tight_layout()
plt.savefig(output_plot)
plt.close()
print(f"📊 Grafico salvato in: {output_plot}")
